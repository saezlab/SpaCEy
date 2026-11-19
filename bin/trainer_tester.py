import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

import custom_tools
from data_processing import augment, prepare_graphs
from dataset import TissueDataset
from eval import risk_concordance_index
from loss import CoxPHLoss, CoxRankLoss, events_from_censored
from model import CustomGCN

SEED = 42
CONFIG_KEYS = ["model", "lr", "bs", "dropout", "epoch", "num_of_gcn_layers", "num_of_ff_layers", "gcn_h", "fcl",
               "weight_decay", "factor", "patience", "min_lr", "aggregators", "scalers", "heads"]


# Training and evaluation steps

def build_model(config: Dict, num_node_features: int, device: str, num_classes: int = 1,
                label_type: str = "regression") -> CustomGCN:
    """Graph model from a hyperparameter dict; one output (a risk score) by default."""
    return CustomGCN(
        type=str(config["model"]), num_node_features=num_node_features,
        num_gcn_layers=int(config["num_of_gcn_layers"]), num_ff_layers=int(config["num_of_ff_layers"]),
        gcn_hidden_neurons=int(config["gcn_h"]), ff_hidden_neurons=int(config["fcl"]),
        dropout=float(config["dropout"]), aggregators=list(config["aggregators"]), scalers=list(config["scalers"]),
        deg=None, heads=int(config["heads"]), num_classes=num_classes, label_type=label_type,
        use_batch_norm=bool(config.get("use_batch_norm", False)), pooling=config.get("pooling", "mean"),
        edge_dim=config.get("edge_dim"),
    ).to(device)


def survival_events(batch) -> torch.Tensor:
    return events_from_censored(batch.is_censored.view(-1).float())


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None)).float()
        loss = criterion(out, batch.y, survival_events(batch))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += float(loss.item())
    return total / max(len(loader), 1)


@torch.no_grad()
def predict(model, loader, criterion, device) -> Tuple[float, pd.DataFrame]:
    """Loss and one row per graph: image and patient id, risk, survival time, censoring flag and death."""
    model.eval()
    total, rows = 0.0, []
    for batch in loader:
        img_ids, p_ids = list(batch.img_id), list(batch.p_id)
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, edge_attr=getattr(batch, "edge_attr", None)).float()
        total += float(criterion(out, batch.y, survival_events(batch)).item())
        for img, pid, risk, os_m, cens in zip(img_ids, p_ids, out.view(-1).cpu().numpy(),
                                              batch.osmonth.view(-1).cpu().numpy(),
                                              batch.is_censored.view(-1).cpu().numpy()):
            rows.append({"img_id": str(img), "p_id": str(pid), "risk": float(risk), "os_month": float(os_m),
                         "is_censored": int(cens), "death": 1 - int(cens)})
    return total / max(len(loader), 1), pd.DataFrame(rows)


def survival_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Image-level and patient-level (mean risk over a patient's images) C-index, event = death."""
    patient = df.groupby("p_id").agg(risk=("risk", "mean"), os_month=("os_month", "first"),
                                     death=("death", "first"))
    return {"graph_ci": risk_concordance_index(df.os_month, df.risk, df.death),
            "patient_ci": risk_concordance_index(patient.os_month, patient.risk, patient.death),
            "n_graphs": len(df), "n_patients": len(patient), "n_deaths_patient": int(patient.death.sum())}


def patient_reduce(log_h: torch.Tensor, pids: List[str]) -> Tuple[torch.Tensor, np.ndarray]:
    codes, _ = pd.factorize(pd.Series(pids))
    idx = torch.tensor(codes.tolist(), device=log_h.device)
    n = int(codes.max()) + 1
    sums = torch.zeros(n, device=log_h.device).index_add(0, idx, log_h)
    counts = torch.zeros(n, device=log_h.device).index_add(0, idx, torch.ones_like(log_h))
    first = np.unique(codes, return_index=True)[1]
    return sums / counts, first


def forward_all(model, loader, device):
    outs, pids, durations, censored = [], [], [], []
    for batch in loader:
        pids += [str(p) for p in batch.p_id]
        batch = batch.to(device)
        outs.append(model(batch.x, batch.edge_index, batch.batch,
                          edge_attr=getattr(batch, "edge_attr", None)).float().view(-1))
        durations.append(batch.y.view(-1).float())
        censored.append(batch.is_censored.view(-1).float())
    return torch.cat(outs), pids, torch.cat(durations), torch.cat(censored)


def patient_loss(model, loader, device, criterion):
    log_h, pids, durations, censored = forward_all(model, loader, device)
    risk, first = patient_reduce(log_h, pids)
    first = torch.tensor(first.tolist(), device=log_h.device)
    return criterion(risk, durations[first], 1.0 - censored[first])


def train_epoch_aug(model, loader, optimizer, criterion, device) -> None:
    model.train()
    for batch in loader:
        batch = batch.to(device)
        x, edge_index, graph_index, edge_attr = augment(batch)
        out = model(x, edge_index, graph_index, edge_attr=edge_attr).float()
        loss = criterion(out, batch.y, 1.0 - batch.is_censored.view(-1).float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def fit_graph(config, graphs, train_idx, val_idx, device, training, seed, select="ci"):

    custom_tools.set_seeds(seed, deterministic=False)
    criterion = CoxRankLoss(float(config.get("rank_weight", 0.0)))
    model = build_model(config, graphs[0].num_node_features, device)
    custom_tools.set_seeds(seed, deterministic=False)  # model construction must not fix the batch order
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=config["factor"], patience=config["patience"],
                                  min_lr=config["min_lr"])
    bs = int(config["bs"]) if training.startswith("mini") else 64
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=bs, shuffle=True,
                              drop_last=bool(config.get("use_batch_norm", False)))
    val_loader = DataLoader([graphs[i] for i in val_idx], batch_size=64, shuffle=False)

    best_loss, best_score, best_state, best_epoch, counter = math.inf, -math.inf, None, -1, 0
    lr_history = []
    min_epochs = int(config.get("min_epochs", 0))  # the chosen checkpoint has at least this many epochs
    for epoch in range(int(config["epoch"])):
        lr_history.append(optimizer.param_groups[0]["lr"])
        model.train()
        if training == "mini":
            train_epoch(model, train_loader, optimizer, criterion, device)
        elif training == "mini_aug":
            train_epoch_aug(model, train_loader, optimizer, criterion, device)
        else:
            optimizer.zero_grad()
            loss = patient_loss(model, train_loader, device, criterion)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(patient_loss(model, val_loader, device, criterion))
        _, val_df = predict(model, val_loader, criterion, device)
        score = survival_metrics(val_df)["patient_ci"]
        scheduler.step(val_loss)
        if epoch + 1 < min_epochs:  # too early to be chosen or to stop
            continue
        if (select == "ci" and score > best_score) or (select == "loss" and val_loss < best_loss):
            best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        if val_loss < best_loss:
            best_loss, counter = val_loss, 0
        else:
            counter += 1
            if counter >= 2 * int(config["patience"]):
                break
    model.load_state_dict(best_state)
    return model, best_score, best_epoch, lr_history


def refit_graph(config, graphs, train_idx, device, seed, lr_history, training="mini"):
    """Retrain on all training patients, replaying the learning-rate schedule up to the selected epoch."""
    custom_tools.set_seeds(seed, deterministic=False)
    criterion = CoxRankLoss(float(config.get("rank_weight", 0.0)))
    model = build_model(config, graphs[0].num_node_features, device)
    custom_tools.set_seeds(seed, deterministic=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=int(config["bs"]), shuffle=True,
                              drop_last=bool(config.get("use_batch_norm", False)))
    for lr in lr_history:
        for group in optimizer.param_groups:
            group["lr"] = lr
        if training == "mini_aug":
            train_epoch_aug(model, train_loader, optimizer, criterion, device)
        else:
            train_epoch(model, train_loader, optimizer, criterion, device)
    return model


def graph_patient_risk(model, graphs, idx, device) -> pd.DataFrame:
    loader = DataLoader([graphs[i] for i in idx], batch_size=64, shuffle=False)
    _, df = predict(model, loader, CoxPHLoss(), device)
    return df


class trainer_tester:

    def __init__(self, parser_args, setup_args) -> None:
        """Runs the survival cross-validation, or with --full_training trains one model on all patients.

        Args:
            parser_args (Namespace): Holds the arguments that came from parsing CLI
            setup_args (Namespace): Holds the arguments that came from setup
        """
        custom_tools.set_seeds(seed=SEED, deterministic=False)
        self.parser_args = parser_args
        self.setup_args = setup_args
        self.set_device()
        if getattr(parser_args, "full_training", False):
            self.full_train_loop()
        else:
            self.cv_loop()

    def set_device(self):
        """Sets up the computation device for the class
        """
        self.device = custom_tools.get_device(self.parser_args.gpu_id)
        if self.device == "cpu":
            raise RuntimeError("no GPU found!")


    def get_config(self) -> dict:
        """Hyperparameters from the CLI; a --params_json file (key "hyperparameters") overrides them."""
        args = self.parser_args
        config = {k: getattr(args, k) for k in CONFIG_KEYS}
        config.update(use_batch_norm=False, pooling="mean", augment=args.augment)
        if args.params_json:
            config.update(json.loads(Path(args.params_json).read_text())["hyperparameters"])
        config["min_epochs"] = args.min_epochs
        return config

    def cv_loop(self) -> None:
        args = self.parser_args
        if not args.folds_json:
            raise ValueError("give the cross-validation folds with --folds_json")
        config = self.get_config()
        training = "mini_aug" if config["augment"] else "mini"

        dataset = TissueDataset(os.path.join(self.setup_args.S_PATH, f"../data/{args.dataset_name}", args.unit),
                                args.unit)
        meta = custom_tools.survival_metadata(dataset)
        raw_graphs = [dataset[i] for i in range(len(dataset))]
        folds = custom_tools.load_folds_json(args.folds_json, meta)
        print(f"cv_setting {args.cv_setting}: {len(dataset)} images, {meta.p_id.nunique()} "
              f"patients, {len(folds)} folds, feature_normalise {args.feature_normalise}")
        print("config:", config)

        rows, preds = [], []
        for fold in sorted(folds):
            seed = getattr(args, "seed", SEED) + fold
            train_idx, test_idx = folds[fold]
            full = prepare_graphs(raw_graphs, train_idx, args.feature_normalise)
            if args.cv_setting == "nfold":
                model, _, best_epoch, _ = fit_graph(config, full, train_idx, test_idx, self.device, training,
                                                        seed, select="ci")
            else:
                inner_train, inner_val = custom_tools.inner_patient_split(meta, train_idx, 0.2, seed)
                graphs = prepare_graphs(raw_graphs, inner_train, args.feature_normalise)
                model, _, best_epoch, lr_history = fit_graph(config, graphs, inner_train, inner_val,
                                                                 self.device, training, seed, select="loss")
                model = refit_graph(config, full, train_idx, self.device, seed, lr_history[: best_epoch + 1],
                                        training)
            df = graph_patient_risk(model, full, test_idx, self.device)
            patient = df.groupby("p_id").agg(risk=("risk", "mean"), os_month=("os_month", "first"),
                                             death=("death", "first"))
            ci = risk_concordance_index(patient.os_month, patient.risk, patient.death)
            rows.append({"fold": fold, "test_patient_ci": ci, "test_image_ci": risk_concordance_index(df.os_month, df.risk,
                                                                                            df.death),
                         "n_test_patients": len(patient),
                         "n_test_events": int(patient.death.sum())})
            preds.append(df.assign(fold=fold))
            torch.save(model.state_dict(), os.path.join(self.setup_args.MODEL_PATH,
                                                        f"{self.setup_args.id}_model_fold{fold}.pt"))
            print(f"fold {fold}: test_ci {ci:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
        # TODO: make this more informative later
        result = pd.DataFrame(rows)
        out = Path(self.setup_args.RESULT_PATH)
        result.to_csv(out / f"{self.setup_args.id}_fold_c_index.csv", index=False)
        pd.concat(preds).to_csv(out / f"{self.setup_args.id}_test_predictions.csv", index=False)
        (out / f"{self.setup_args.id}_config.json").write_text(json.dumps(
            {"config": config, "folds_json": str(args.folds_json), "cv_setting": args.cv_setting,
             "feature_normalise": args.feature_normalise, "seed": f"{getattr(args, 'seed', SEED)} + fold",
             "mean_test_patient_ci": result.test_patient_ci.mean()}, indent=1))
        print(result[["fold", "test_patient_ci"]].rename(columns={"test_patient_ci": "test_ci"}).round(4).to_string(index=False))
        print(f"mean test_ci {result.test_patient_ci.mean():.4f} [cv_setting {args.cv_setting}] "
              f"(SD {result.test_patient_ci.std():.4f}); results in {out}")

    def full_train_loop(self) -> None:
        
        args = self.parser_args
        config = self.get_config()
        training = "mini_aug" if config["augment"] else "mini"
        seed = getattr(args, "seed", SEED)

        dataset = TissueDataset(os.path.join(self.setup_args.S_PATH, f"../data/{args.dataset_name}", args.unit),
                                args.unit)
        meta = custom_tools.survival_metadata(dataset)
        raw_graphs = [dataset[i] for i in range(len(dataset))]
        all_idx = list(range(len(raw_graphs)))
        print(f"full training: {len(dataset)} images, {meta.p_id.nunique()} patients, "
              f"feature_normalise {args.feature_normalise}")
        print("config:", config)

        inner_train, inner_val = custom_tools.inner_patient_split(meta, all_idx, 0.2, seed)
        graphs = prepare_graphs(raw_graphs, inner_train, args.feature_normalise)
        _, _, best_epoch, lr_history = fit_graph(config, graphs, inner_train, inner_val, self.device, training, seed,
                                                 select="loss")
        n_epochs = best_epoch + 1
        full = prepare_graphs(raw_graphs, all_idx, args.feature_normalise)
        model = refit_graph(config, full, all_idx, self.device, seed, lr_history[:n_epochs], training)

        df = graph_patient_risk(model, full, all_idx, self.device)
        patient = df.groupby("p_id").agg(risk=("risk", "mean"), os_month=("os_month", "first"),
                                         death=("death", "first"))
        train_ci = risk_concordance_index(patient.os_month, patient.risk, patient.death)

        settings = {**vars(args), **config, "num_node_features": int(raw_graphs[0].num_node_features),
                    "train_patient_ci": train_ci, "seed": seed,
                    "n_patients": int(meta.p_id.nunique()), "n_images": len(raw_graphs)}
        if args.feature_normalise == "log1p_z":  # the statistics new graphs must be scaled with
            x = torch.log1p(torch.cat([g.x for g in raw_graphs]))
            settings["log1p_z_mean"] = x.mean(0).tolist()
            settings["log1p_z_std"] = x.std(0).clamp_min(1e-6).tolist()
        custom_tools.save_model(model=model, fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
        custom_tools.save_dict_as_json(settings, self.setup_args.id, self.setup_args.MODEL_PATH)
        out = Path(self.setup_args.RESULT_PATH)
        df.to_csv(out / f"{self.setup_args.id}_train_predictions.csv", index=False)
        print(f"training-set patient C-index {train_ci:.4f} (in-sample, not a performance estimate); "
              f"model in {self.setup_args.MODEL_PATH}")
