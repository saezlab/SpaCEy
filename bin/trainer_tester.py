from cProfile import label
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
import torch
from model import CustomGCN
from dataset import TissueDataset, LungDataset
from torch_geometric.loader import DataLoader
import numpy as np
import torch.nn as nn
import plotting
from tqdm import tqdm
import pandas as pd
import os
import random
from torch_geometric.utils import degree
from evaluation_metrics import r_squared_score, mse, rmse, mae
import custom_tools
import csv
import statistics
import seaborn as sns

import pytorch_lightning as pl
from eval import concordance_index
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import PNAConv 
from early_stopping import EarlyStopping

# torch.autograd.set_detect_anomaly(True)


class trainer_tester:


    def __init__(self, parser_args, setup_args ) -> None:
        """Manager of processes under train_tester, main goel is to show the processes squentially

        Args:
            parser_args (Namespace): Holds the arguments that came from parsing CLI
            setup_args (Namespace): Holds the arguments that came from setup
        """
        custom_tools.set_seeds(seed=42, deterministic=False)
        self.parser_args = parser_args
        self.setup_args = setup_args
        self.set_device()
        self.init_folds()

        if self.parser_args.full_training:
            self.full_train_loop()
        elif self.parser_args.t_v_t:
            self.train_val_test_loop()
        else:
            self.train_test_loop()

        # self.save_results()

    def set_device(self):
        """Sets up the computation device for the class
        """
        self.device = custom_tools.get_device(self.parser_args.gpu_id)

    def convert_to_month(self, df_col):
        if self.parser_args.unit=="week":
            # convert to month
            return df_col/4.0
        elif self.parser_args.unit=="month":
            return df_col
        elif self.parser_args.unit=="week_lognorm":
            return np.exp(df_col)/4.0
        else:
            raise Exception("Invalid target unit... Should be week,  month, or week_lognorm")

    def init_folds(self):
        """Pulls data, creates samplers according to ratios, creates train, test and validation loaders for 
        each fold, saves them under the 'folds_dict' dictionary
        """
        if self.parser_args.dataset_name == "Lung":
            self.dataset = LungDataset(os.path.join(self.setup_args.S_PATH, f"../data/{self.parser_args.dataset_name}"), self.parser_args.label)
        else:
            self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH, f"../data/{self.parser_args.dataset_name}", self.parser_args.unit),  self.parser_args.unit)

        print("Number of samples:", len(self.dataset), self.parser_args.dataset_name,  self.parser_args.label)

        if self.parser_args.label == "OSMonth" or self.parser_args.loss == "CoxPHLoss":
            self.label_type = "regression"
            self.num_classes = 1

        elif self.parser_args.label == "tumor_grade":
            self.label_type = "classification"
            tumor_grade_raw = self.dataset.data.y
            tumor_grade_tensor = torch.tensor(tumor_grade_raw) if not isinstance(tumor_grade_raw, torch.Tensor) else tumor_grade_raw
            self.num_classes = int(tumor_grade_tensor.max().item()) + 1
            
            class_counts = torch.bincount(tumor_grade_tensor.long(), minlength=self.num_classes)
            # Avoid division by zero: set zero counts to 1 for weight calculation, then set their weight to 0
            safe_class_counts = class_counts.clone()
            safe_class_counts[safe_class_counts == 0] = 1
            class_weights = 1.0 / safe_class_counts.float()
            class_weights[class_counts == 0] = 0.0
            class_weights = class_weights[:self.num_classes]
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            if isinstance(class_weights, torch.Tensor):
                class_weights = class_weights.to(self.device)
            self.setup_args.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            self.dataset.data.y = tumor_grade_tensor
            self.uniq_tumor_grades = sorted(list(set(tumor_grade_tensor.tolist())))
            print("Number of classes:", self.num_classes)

        elif self.parser_args.label == "DiseaseStage":
            self.label_type = "classification"
            # Only set class_weights if defined for DiseaseStage
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()

        elif self.parser_args.label in ["Relapse","Progression"]:
            print("Classification")
            self.label_type = "classification"
            # self.setup_args.criterion = torch.nn.CrossEntropyLoss()
            self.dataset.data.y = self.dataset.data.y
            self.unique_classes = torch.unique(self.dataset.data.y)
            # self.num_classes = len(self.unique_classes)
            self.num_classes = 1

        self.label_data = self.dataset.data.y
        self.dataset = self.dataset.shuffle()

        self.fold_dicts = []
        deg = -1

        if self.parser_args.full_training:
            # TODO: Consider bootstrapping 
            train_sampler = torch.utils.data.SubsetRandomSampler(list(range(len(self.dataset))))
            train_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, shuffle=True)
            validation_loader, test_loader = None, None
            if self.parser_args.model in ["PNAConv", "MMAConv", "GMNConv"]:
                    deg = self.calculate_deg(train_sampler)

            model = self.set_model(deg)
            

            optimizer = torch.optim.Adam(model.parameters(), lr=self.parser_args.lr, weight_decay=self.parser_args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor= self.parser_args.factor, patience=self.parser_args.patience, min_lr=self.parser_args.min_lr, verbose=True)

            fold_dict = {
                    "fold": 1,
                    "train_loader": train_loader,
                    "validation_loader": validation_loader,
                    "test_loader": test_loader,
                    "deg": deg,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler
                }

            self.fold_dicts.append(fold_dict)

        # Add train/validation/test split if t_v_t is True
        elif self.parser_args.t_v_t:
            print("Creating train/validation/test split")
            # Create train/validation/test split
            dataset_size = len(self.dataset)
            train_size = int(0.7 * dataset_size)
            val_size = int(0.15 * dataset_size)
            test_size = dataset_size - train_size - val_size
            print("Train size:", train_size, "Validation size:", val_size, "Test size:", test_size)
            
            # Create indices for the split
            indices = list(range(dataset_size))
            random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create samplers
            train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
            test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
            
            # Create data loaders
            train_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler=train_sampler)
            validation_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler=val_sampler)
            test_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler=test_sampler)
            
            # Calculate degree if needed
            deg = -1
            if self.parser_args.model in ["PNAConv", "MMAConv", "GMNConv"]:
                deg = self.calculate_deg(train_sampler)
            
            # Create model, optimizer, and scheduler
            model = self.set_model(deg)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.parser_args.lr, weight_decay=self.parser_args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=self.parser_args.factor, patience=self.parser_args.patience, min_lr=self.parser_args.min_lr, verbose=True)
            
            # Create fold dict for train/val/test split
            tvt_fold_dict = {
                "fold": "train_val_test",
                "train_loader": train_loader,
                "validation_loader": validation_loader,
                "test_loader": test_loader,
                "deg": deg,
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler
            }
            
            self.fold_dicts.append(tvt_fold_dict)

        
        else:
            
            # self.samplers = custom_tools.k_fold_by_group(self.dataset)
            if self.parser_args.dataset_name == "Lung":
                self.samplers = custom_tools.create_stratified_cv_folds_lung_dataset(self.dataset, self.parser_args.label)
            else:
                self.samplers = custom_tools.get_n_fold_split(self.dataset, self.parser_args.dataset_name)

            deg = -1

            self.parser_args.fold_img_id_dict = dict()
            for fold, train_sampler, validation_sampler in self.samplers:
                # print("Creating k folds", fold, train_sampler.ids)
                train_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= train_sampler)
                validation_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= validation_sampler)
                # test_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= test_sampler)
                
                train_img_ids = []
                val_img_ids = []
                for data in train_loader:
                    train_img_ids.extend(data.img_id)
                for data in validation_loader:
                    val_img_ids.extend(data.img_id)

                self.parser_args.fold_img_id_dict[f"fold_{fold}"] = [train_img_ids, val_img_ids]

                if self.parser_args.model in ["PNAConv", "MMAConv", "GMNConv"]:
                    deg = self.calculate_deg(train_sampler)
                
                model = self.set_model(deg)
                # print("Model", model)

                optimizer = torch.optim.Adam(model.parameters(), lr=self.parser_args.lr, weight_decay=self.parser_args.weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, 'min', factor= self.parser_args.factor, patience=self.parser_args.patience, min_lr=self.parser_args.min_lr, verbose=True)

                fold_dict = {
                    "fold": fold,
                    "train_loader": train_loader,
                    "validation_loader": validation_loader,
                    "deg": deg,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler
                }

                self.fold_dicts.append(fold_dict)
            # print(self.parser_args.fold_img_id_dict)
    
    def calculate_deg(self, train_sampler):
        """Calcualtes deg, which is necessary for some models

        Args:
            train_sampler (_type_): Training data sampler
        """
        train_dataset = self.dataset[train_sampler.indices]
        train_loader = DataLoader(train_dataset, batch_size=self.parser_args.bs, shuffle=True)
        deg = PNAConv.get_degree_histogram(train_loader)

        return deg

    def set_model(self, deg):
        """Sets the model according to parser parameters and deg

        Args:
            deg (_type_): degree data of the graph
        """
        # print(vars(self.parser_args))#  = self.dataset.num_node_features
        self.parser_args.num_node_features = self.dataset.num_node_features
        model = CustomGCN(
                    type = self.parser_args.model,
                    num_node_features = self.dataset.num_node_features, ####### LOOOOOOOOK HEREEEEEEEEE
                    num_gcn_layers=self.parser_args.num_of_gcn_layers, 
                    num_ff_layers=self.parser_args.num_of_ff_layers, 
                    gcn_hidden_neurons=self.parser_args.gcn_h, 
                    ff_hidden_neurons=self.parser_args.fcl, 
                    dropout=self.parser_args.dropout,
                    aggregators=self.parser_args.aggregators,
                    scalers=self.parser_args.scalers,
                    deg = deg, # Comes from data not hyperparameter
                    num_classes = self.num_classes,
                    heads = self.parser_args.heads,
                    label_type = self.label_type
                        ).to(self.device)

        return model
        

    def train(self, fold_dict):
        """Trains the network

        Args:
            fold_dict (dict): Holds data about the used fold

        Returns:
            float: Total loss
        """
        fold_dict["model"].train()
        total_loss = 0.0
        pred_list = []
        # WARN Disabled it but IDK what it does
        #out_list = []
        # print(fold_dict["model"])
        for data in fold_dict["train_loader"]:  # Iterate in batches over the training dataset.

            out = fold_dict["model"](data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).float().to(self.device) # Perform a single forward pass.
            # print("Out shape", out.shape, torch.sigmoid(out).shape, data.y.shape)
            loss = None
            if self.parser_args.loss == "CoxPHLoss":#  or self.parser_args.loss == "NegativeLogLikelihood":
                loss = self.setup_args.criterion(out, data.y.to(self.device), data.is_censored.to(self.device))  # Compute the loss.    
            else:
                target = data.y
                if self.label_type == "classification":
                    target = data.y.long()
                    if target.dim() > 1:
                        target = target.view(-1)
                    # Do NOT squeeze the output for classification
                    if self.num_classes == 1:
                        loss = self.setup_args.criterion(out, target.to(self.device).float().unsqueeze(1))
                    else:
                        loss = self.setup_args.criterion(out, data.y.long().to(self.device))
                else:
                    # For regression, squeeze output and target
                    loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))
                # print("Loss", loss, loss.item(), data.y.to(self.device), out.squeeze(), out)
            
            loss.backward()  # Derive gradients.
            
            if self.label_type == "classification":
                preds = out.argmax(dim=1)
                pred_list.extend(preds.cpu().numpy().tolist())
            else:
                pred_list.extend([val.item() for val in out.squeeze()])
            total_loss += float(loss.item())
            fold_dict["optimizer"].step()  # Update parameters based on gradients.
            fold_dict["optimizer"].zero_grad()  # Clear gradients

            # Print mean absolute gradient for all parameters after backward
            total_grad = 0.0
            n_params = 0
            for name, param in fold_dict["model"].named_parameters():
                if param.grad is not None:
                    grad_abs = param.grad.abs().mean().item()
                    print(f"[Grad] {name}: mean abs grad = {grad_abs}")
                    total_grad += grad_abs
                    n_params += 1
            if n_params > 0:
                print(f"[Grad] Mean abs grad across all params: {total_grad / n_params}")

        return total_loss

    def test(self, model, loader, fold, label=None, return_pred_df=False):
        """Tests the model on wanted loader


        Args:
            fold_dict (dict): Holds data about the used fold
            test_on (str): Which loader to test on (train_loader, test_loader, valid_loader)
            label (_type_, optional): Label of the loader. Defaults to None.
            plot_pred (bool, optional): Should there be a plot. Defaults to False.

        Returns:
            float: total loss
        """

        # loader = fold_dict[test_on]
        model.eval()

        total_loss = 0.0
        pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list, censorship_list, progression_list, sample_id_list, disease_stage_list = [], [], [], [], [], [], [], [], [], [], []
        
        for data in loader:  # Iterate in batches over the training/test dataset.
            if data.y.shape[0]>1:
                
                out = model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).float().to(self.device) # Perform a single forward pass.
                # print("out", out)
                # print("data.y", data.y)
                loss = None
                if self.parser_args.loss == "CoxPHLoss":#  or self.parser_args.loss=="NegativeLogLikelihood":
                    loss = self.setup_args.criterion(out, data.y.to(self.device), data.is_censored.to(self.device))  # Compute the loss.    
                else:
                    if self.label_type == "classification":
                        preds = out.argmax(dim=1)
                        pred_list.extend(preds.cpu().numpy().tolist())
                        true_list.extend(data.y.cpu().numpy().tolist())
                        
                        if self.num_classes == 1:
                            loss = self.setup_args.criterion(out, data.y.to(self.device).float().unsqueeze(1))
                        else:
                            loss = self.setup_args.criterion(out, data.y.long().to(self.device))

                        # Collect probabilities for AUC
                        if out.shape[1] == 1:
                            # Binary classification
                            probs = torch.sigmoid(out).detach().cpu().numpy().flatten()
                            if not hasattr(self, 'train_probabilities'):
                                self.train_probabilities = []
                            self.train_probabilities.extend(probs.tolist())
                        else:
                            # Multi-class classification
                            probs = torch.softmax(out, dim=1).detach().cpu().numpy()
                            for i in range(out.shape[1]):
                                col_name = str(i)
                                if col_name not in locals():
                                    locals()[col_name] = []
                                if not hasattr(self, 'train_probabilities'):
                                    self.train_probabilities = []
                            if not hasattr(self, 'train_prob_matrix'):
                                self.train_prob_matrix = []
                            self.train_prob_matrix.extend(probs.tolist())
                    else:
                        loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.
                        true_list.extend([round(val.item(),6) for val in data.y])
                        pred_list.extend([val.item() for val in out.squeeze()])
                total_loss += float(loss.item())

                if return_pred_df and self.parser_args.loss == "CoxPHLoss":
                    tumor_grade_list.extend([val for val in data.tumor_grade])
                    clinical_type_list.extend([val for val in data.clinical_type])
                    osmonth_list.extend([val.item() for val in data.osmonth])
                    censorship_list.extend([val.item() for val in data.is_censored])
                    pid_list.extend([val for val in data.p_id])
                    img_list.extend([val for val in data.img_id])
                    
                elif return_pred_df  and self.parser_args.label in ["Relapse","Progression"]:

                    # print(data) osmonth=[64], sample_id=[64], img_id=[64], clinical_type=[64], disease_stage=[64]

                    clinical_type_list.extend([val for val in data.clinical_type])
                    disease_stage_list.extend([val for val in data.disease_stage])
                    progression_list.extend([val.item() for val in data.y])
                    osmonth_list.extend([val.item() for val in data.osmonth])
                    sample_id_list.extend([val for val in data.sample_id])
                    img_list.extend([val for val in data.img_id])
                    

                    """tumor_grade_list.extend([val for val in data.tumor_grade])
                    clinical_type_list.extend([val for val in data.clinical_type])
                    osmonth_list.extend([val.item() for val in data.osmonth])
                    relapse_list.extend([val.item() for val in data.y])
                    pid_list.extend([val for val in data.p_id])
                    img_list.extend([val for val in data.img_id])"""
                    
                    
                
                else:
                    pass
            else:
                pass

                    
        # print("pred_list", pred_list)
        if return_pred_df:
            
            # label_list = [str(fold_dict["fold"]) + "-" + label]*len(clinical_type_list)
            label_list = [str(fold) + "-" + label]*len(clinical_type_list) # 
            if self.parser_args.loss == "CoxPHLoss":
                df = pd.DataFrame(list(zip(pid_list, img_list, true_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list, censorship_list, label_list)),
               columns =["Patient ID","Image Number", 'True Value', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month", "Censored", "Fold#-Set"])
            else:
                # print("Here1")
                # df = pd.DataFrame(list(zip(pid_list, img_list, true_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list,  label_list)),
                # columns =["Patient ID","Image Number", 'True Value', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month", "Fold#-Set"])
                """clinical_type_list.extend([val for val in data.clinical_type])
                    disease_stage_list.extend([val for val in data.disease_stage])
                    progression_list.extend([val.item() for val in data.y])
                    osmonth_list.extend([val.item() for val in data.osmonth])
                    sample_id_list.extend([val for val in data.sample_id])
                    img_list.extend([val for val in data.img_id])
                """
                df = pd.DataFrame(list(zip(sample_id_list, img_list, true_list, pred_list, progression_list, clinical_type_list, osmonth_list,  label_list)), columns =["Sample ID","Image Number", 'True Value', 'Predicted', "Progression", "Clinical Type", "OS Month", "Fold#-Set"])
                # Add probabilities for AUC
                if self.label_type == "classification":
                    if hasattr(self, 'train_probabilities'):
                        if self.num_classes == 1 or self.num_classes == 2:
                            # Binary classification
                            df['Probabilities'] = self.train_probabilities[:len(df)]
                            self.train_probabilities = self.train_probabilities[len(df):]
                        else:
                            # Multi-class classification
                            if hasattr(self, 'train_prob_matrix'):
                                prob_matrix = np.array(self.train_prob_matrix[:len(df)])
                                for i in range(prob_matrix.shape[1]):
                                    df[str(i)] = prob_matrix[:, i]
                                self.train_prob_matrix = self.train_prob_matrix[len(df):]
            return total_loss, df
        else:
            return total_loss

    def train_test_loop(self):
        """
        Training and testing occurs under this function. 
        """
        self.results =[] 
        # collect train/val/test predictions of all folds in all_preds_df
        fold_val_scores = []
        fold_results = []  # To store per-fold metrics

        # print(self.fold_dicts)
        for fold_dict in self.fold_dicts:
            # Print label distribution and unique values of y at the start of training (once per fold)
            if self.label_type == "classification":
                
                y_tensor = self.dataset.data.y if isinstance(self.dataset.data.y, torch.Tensor) else torch.tensor(self.dataset.data.y)
                unique, counts = torch.unique(y_tensor, return_counts=True)
                # print(f"[Zero-indexed] Label distribution for {self.parser_args.label}: {dict(zip(unique.tolist(), counts.tolist()))}")
                # print(f"[Zero-indexed] Unique values in y: {unique.tolist()}")
            # print(fold_dict["model"])
            best_val_loss = np.inf
            early_stopping = EarlyStopping(patience=self.parser_args.patience*2, verbose=True, model_path=self.setup_args.MODEL_PATH)

            print(f"########## Fold :  {fold_dict['fold']} ########## ")
            for epoch in (pbar := tqdm(range(self.parser_args.epoch), disable=False)):

                self.train(fold_dict)
                # Print label distribution at the start of training
                # if self.label_type == "classification" and hasattr(self.dataset.data, self.parser_args.label):
                #     label_tensor = torch.tensor(getattr(self.dataset.data, self.parser_args.label))
                #     unique, counts = torch.unique(label_tensor, return_counts=True)
                #     print(f"Label distribution for {self.parser_args.label}: {dict(zip(unique.tolist(), counts.tolist()))}")
                # Print model outputs and predictions for the first batch of validation set
                model = fold_dict["model"]
                model.eval()

                train_loss = self.test(fold_dict["model"], fold_dict["train_loader"],  fold_dict["fold"])
                validation_loss, df_epoch_val = self.test(fold_dict["model"], fold_dict["validation_loader"],  fold_dict["fold"], "validation", self.setup_args.plot_result)
                #print("train_loss", train_loss, "validation_loss", validation_loss, df_epoch_val)
                # test_loss, df_epoch_test = self.test(fold_dict["model"], fold_dict["test_loader"],  fold_dict["fold"], "test", self.setup_args.plot_result)
                epoch_val_score = 0.0
                auc_score = None
                if self.parser_args.loss == "CoxPHLoss":
                    epoch_val_score = concordance_index(df_epoch_val['OS Month'], -df_epoch_val['Predicted'], df_epoch_val["Censored"]) if self.parser_args.loss=="NegativeLogLikelihood" else concordance_index(df_epoch_val['OS Month'], -df_epoch_val['Predicted'], df_epoch_val["Censored"])
                else:
                    correct = (df_epoch_val['True Value'] == df_epoch_val['Predicted']).sum()
                    accuracy = correct / len(df_epoch_val)
                    # --- AUC Calculation ---
                    if self.label_type == "classification":
                        if self.num_classes == 2 or self.num_classes == 1:
                            # Binary classification
                            
                            y_true = df_epoch_val["True Value"].values
                            
                            y_score = df_epoch_val["Probabilities"].values if "Probabilities" in df_epoch_val else None
                            # print("y_true", y_true, "y_score", y_score)
                            if y_score is None and hasattr(self, 'train_probabilities'):
                                y_score = self.train_probabilities
                            auc_score = roc_auc_score(y_true, y_score)
                        else:
                            # Multi-class classification
                            y_true = df_epoch_val["True Value"].values
                            y_score = df_epoch_val[[str(i) for i in range(self.num_classes)]].values if all(str(i) in df_epoch_val.columns for i in range(self.num_classes)) else None
                            if y_score is not None:
                                auc_score = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
                            else:
                                auc_score = None
                    epoch_val_score = auc_score
                # Print per-epoch metrics
                if self.label_type == "classification":
                    print(f"Epoch {epoch} | Fold {fold_dict['fold']} | Val Loss: {validation_loss:.4f} | Val AUC: {epoch_val_score:.4f} | Val Acc: {accuracy:.4f}")

            
                fold_dict["scheduler"].step(validation_loss)
                
                early_stopping(validation_loss, epoch_val_score, fold_dict["model"], vars(self.parser_args), id_file_name=self.setup_args.id, deg=self.fold_dicts[0]["deg"] if self.parser_args.model in ["PNAConv", "MMAConv", "GMNConv"] else None)
                # pbar.set_description(f"Train loss: {train_loss:.2f} Val. loss: {validation_loss:.2f} Val c_index: {epoch_val_ci_score} Patience: {early_stopping.counter}")
                pbar.set_description(f"Train loss: {train_loss:.2f} Val. loss: {validation_loss:.2f} Best val. score: {early_stopping.best_eval_score} Patience: {early_stopping.counter}")

                if early_stopping.early_stop or epoch==self.parser_args.epoch-1:
                    # Save best metrics for this fold
                    best_fold_loss = early_stopping.val_loss_min
                    best_fold_score = early_stopping.best_eval_score
                    # Try to get best AUC for this fold (last epoch's AUC)
                    best_fold_auc = auc_score if 'auc_score' in locals() else None
                    fold_results.append([
                        fold_dict['fold'],
                        best_fold_loss,
                        best_fold_score,
                        best_fold_auc
                    ])
                    print("Best model lr:", fold_dict["optimizer"].param_groups[0]["lr"])
                    self.parser_args.best_epoch = epoch
                    fold_val_scores.append(early_stopping.best_eval_score)
                    print("Early stopping the training...")
                    break
                # break
            break

        average_val_scores = sum(fold_val_scores)/len(fold_val_scores)
        print(f"Average validation score: {sum(fold_val_scores)/len(fold_val_scores)}")
        # if average_val_scores > 0.65:
        self.parser_args.ci_score = average_val_scores
        self.parser_args.fold_ci_scores = fold_val_scores
        custom_tools.save_dict_as_json(vars(self.parser_args), self.setup_args.id, self.setup_args.MODEL_PATH)
        # print(f"Average validation score: {sum(fold_val_scores)/len(fold_val_scores)}")
        box_plt = sns.boxplot(data=fold_val_scores)
        fig = box_plt.get_figure()
        fig.savefig(os.path.join(self.setup_args.PLOT_PATH, f"{self.setup_args.id}.png"))
        # Save per-fold results to CSV
        with open(os.path.join(self.setup_args.RESULT_PATH, f"{self.setup_args.id}_per_fold_results.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Fold", "Best Val Loss", "Best Val Accuracy", "Best Val AUC"])
            writer.writerows(fold_results)
        # Plot per-fold best AUCs as a boxplot
        
        import matplotlib.pyplot as plt
        aucs = [row[3] for row in fold_results if row[3] is not None]
        if len(aucs) > 0:
            plt.figure(figsize=(8, 5))
            plt.boxplot(aucs)
            plt.title('Per-Fold Best Validation AUC')
            plt.ylabel('AUC')
            plt.xlabel('Fold')
            plt.savefig(os.path.join(self.setup_args.RESULT_PATH, f"{self.setup_args.id}_per_fold_auc.png"))
            plt.close()
        
    
    
    def full_train_loop(self):
        """Training and testing occurs under this function. 
        """

        self.results =[] 
        # collect train/val/test predictions of all folds in all_preds_df
        all_preds_df = []
        fold_dict = self.fold_dicts[0]    
        best_val_loss = np.inf
        
        print(f"Performing full training ...")
        for epoch in (pbar := tqdm(range(self.parser_args.epoch), disable=True)):

            self.train(fold_dict)

            # train_loss = self.test(fold_dict["model"], "train_loader", 0)
            train_loss = self.test(fold_dict["model"], fold_dict["train_loader"],  fold_dict["fold"])
            pbar.set_description(f"Train loss: {train_loss}")

            if (epoch % self.setup_args.print_every_epoch) == 0:
                print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}')

        train_loss, df_train = self.test(fold_dict["model"], fold_dict["train_loader"], fold_dict["fold"], "train", self.setup_args.plot_result)
        # best_test_loss, df_test = self.test(best_model, fold_dict["test_loader"], fold_dict["fold"], "test", self.setup_args.plot_result) # type: ignore

        if self.label_type == "regression" and self.parser_args.loss == "CoxPHLoss":
            ci_score = concordance_index(df_train['OS Month'], -df_train['Predicted'], df_train["Censored"])

        elif self.label_type == "regression":
            ci_score = concordance_index(df_train['OS Month'], -df_train['Predicted'], df_train["Censored"])
            r2_score = r_squared_score(df_train['True Value'], df_train['Predicted'])
            mse_score = mse(df_train['True Value'], df_train['Predicted'])
            rmse_score = rmse(df_train['True Value'], df_train['Predicted'])

        elif self.label_type == "classification":
            accuracy_Score = accuracy_score(df_train["True Value"], df_train['Predicted'])
            precision_Score = precision_score(df_train["True Value"], df_train['Predicted'],average="micro")   
            f1_Score = f1_score(df_train["True Value"], df_train['Predicted'],average="micro")    
            # --- AUC Calculation ---
            if self.label_type == "classification":
                if self.num_classes == 2 or self.num_classes == 1:
                    # Binary classification
                    y_true = df_train["True Value"].values
                    y_score = df_train["Probabilities"].values if "Probabilities" in df_train else None
                    if y_score is None and hasattr(self, 'train_probabilities'):
                        y_score = self.train_probabilities
                    auc_score = roc_auc_score(y_true, y_score)
                else:
                    # Multi-class classification
                    y_true = df_train["True Value"].values
                    y_score = df_train[[str(i) for i in range(self.num_classes)]].values if all(str(i) in df_train.columns for i in range(self.num_classes)) else None
                    if y_score is not None:
                        auc_score = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
                    else:
                        auc_score = None
            print(f"AUC: {auc_score}")

        all_preds_df = df_train
        

        if self.label_type == "regression" and self.parser_args.loss == "CoxPHLoss":
            self.results.append([fold_dict['fold'], round(100000, 4), round(100000, 4), round(100000, 4), ci_score])

        elif self.label_type == "regression": 
            self.results.append([fold_dict['fold'], round(100000, 4), round(100000, 4), round(100000, 4), r2_score, mse_score, rmse_score])
        
        elif self.label_type == "classification":
            self.results.append([fold_dict['fold'], best_train_loss, best_val_loss, best_test_loss, accuracy_Score, precision_Score, f1_Score])

        # print("Train ci score", ci_score)
        # print(self.label_type, self.parser_args.loss)
        if  (self.label_type == "regression") and self.parser_args.loss == "CoxPHLoss":
            self.parser_args.ci_score = ci_score
            custom_tools.save_dict_as_json(vars(self.parser_args), self.setup_args.id, self.setup_args.MODEL_PATH)
            if not self.parser_args.fold:
                custom_tools.save_model(model=self.fold_dicts[0]["model"], fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
                if self.parser_args.model == "PNAConv":
                    custom_tools.save_pickle(self.fold_dicts[0]["deg"], f"{self.setup_args.id}_deg.pckl", self.setup_args.MODEL_PATH)

        elif  (self.label_type == "regression"):
            # plotting.plot_pred_vs_real(all_preds_df, self.parser_args.en, self.setup_args.id, full_training=True)
            all_preds_df.to_csv(os.path.join(self.setup_args.RESULT_PATH, f"{self.setup_args.id}.csv"), index=False)
            self.save_results()
            custom_tools.save_dict_as_json(vars(self.parser_args), self.setup_args.id, self.setup_args.MODEL_PATH)
            
            if not self.parser_args.fold:
                custom_tools.save_model(model=self.fold_dicts[0]["model"], fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
                if self.parser_args.model == "PNAConv":
                    custom_tools.save_pickle(self.fold_dicts[0]["deg"], f"{self.setup_args.id}_deg.pckl", self.setup_args.MODEL_PATH)
                
    
    def save_results(self):
        """Found results are saved into CSV file
        """
        # header = ["fold number", "best train loss", "best val loss", "best test loss", "fold val r2 score", "fold val mse", "fold val rmse"]
        

        train_results = []
        valid_results = []
        test_results = []
        ci_results = []
        r2_results = []
        mse_results = []
        rmse_results = []
        accuracy_results =[] 
        precision_results =[] 
        f1_results =[] 

        
        if self.label_type == "regression" and self.parser_args.loss=="CoxPHLoss":
            for _,train,valid,test, cindex in self.results:
                train_results.append(train)
                valid_results.append(valid)
                test_results.append(test)
                ci_results.append(cindex)

        elif self.label_type == "regression":
            for _,train,valid,test,r2,mse,rmse in self.results:
                train_results.append(train)
                valid_results.append(valid)
                test_results.append(test)
                r2_results.append(r2)
                mse_results.append(mse)
                rmse_results.append(rmse)

        elif self.label_type == "classification":
            for _,train,valid,test,accuracy,precision,f1 in self.results:
                train_results.append(train)
                valid_results.append(valid)
                test_results.append(test)

                accuracy_results.append(accuracy)
                precision_results.append(precision)
                f1_results.append(f1)

        if self.label_type == "regression":

            header = ["fold_number","train","validation","test","r2","mse","rmse"]

            if self.setup_args.use_fold:
                means = [["Mean", round(statistics.mean(train_results), 4), round(statistics.mean(valid_results), 4), round(statistics.mean(test_results), 4), statistics.mean(r2_results), statistics.mean(mse_results),statistics.mean(rmse_results)]]
                variances = [["Variance", round(statistics.variance(train_results), 4) ,round(statistics.variance(valid_results), 4), round(statistics.variance(test_results), 4), statistics.variance(r2_results),statistics.variance(mse_results),statistics.variance(rmse_results)]]

            else:

                means = [["Mean", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4),r2_results[0],mse_results[0],rmse_results[0]]]
                variances = [["Variance", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4), r2_results[0],mse_results[0],rmse_results[0]]]
        
        elif self.label_type == "classification":

            header = ["fold_number","train","validation","accuracy","precision","f1"]
            if self.setup_args.use_fold:
                means = [["Mean", round(statistics.mean(train_results), 4), round(statistics.mean(valid_results), 4), round(statistics.mean(test_results), 4), statistics.mean(accuracy_results), statistics.mean(precision_results), statistics.mean(f1_results)]]
                variances = [["Variance", round(statistics.variance(train_results), 4) ,round(statistics.variance(valid_results), 4), round(statistics.variance(test_results), 4), statistics.variance(accuracy_results), statistics.mean(precision_results), statistics.mean(f1_results)]]

            else:

                means = [["Mean", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4),accuracy_results[0],precision_results[0],f1_results[0]]]
                variances = [["Variance", round(train_results[0], 4), round(valid_results[0], 4), round(test_results[0], 4), accuracy_results[0],precision_results[0],f1_results[0]]]

        ff = open(os.path.join(self.setup_args.RESULT_PATH, f"{str(self.setup_args.id)}.csv"), 'w')
        ff.close()

        with open(os.path.join(self.setup_args.RESULT_PATH, f"{str(self.setup_args.id)}.csv"), 'w', encoding="UTF8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.results)
            writer.writerows(means)
            writer.writerows(variances)

    def train_val_test_loop(self):
        """
        Training, validation, and testing occurs under this function for train/val/test split.
        """
        self.results = []
        fold_val_scores = []
        fold_results = []

        # Get the train/val/test fold dict (should be the last one added)
        fold_dict = self.fold_dicts[-1]  # Get the train/val/test split
        
        best_val_loss = np.inf
        early_stopping = EarlyStopping(patience=self.parser_args.patience*2, verbose=True, model_path=self.setup_args.MODEL_PATH)

        print(f"########## Train/Val/Test Split Training ##########")
        
        for epoch in (pbar := tqdm(range(self.parser_args.epoch), disable=False)):
            self.train(fold_dict)
            
            model = fold_dict["model"]
            model.eval()

            # Calculate losses for all three splits
            train_loss = self.test(fold_dict["model"], fold_dict["train_loader"], fold_dict["fold"])
            validation_loss, df_epoch_val = self.test(fold_dict["model"], fold_dict["validation_loader"], fold_dict["fold"], "validation", self.setup_args.plot_result)
            test_loss, df_epoch_test = self.test(fold_dict["model"], fold_dict["test_loader"], fold_dict["fold"], "test", self.setup_args.plot_result)
            
            # Calculate validation score
            epoch_val_score = 0.0
            auc_score = None
            if self.parser_args.loss == "CoxPHLoss":
                epoch_val_score = concordance_index(df_epoch_val['OS Month'], -df_epoch_val['Predicted'], df_epoch_val["Censored"]) if self.parser_args.loss=="NegativeLogLikelihood" else concordance_index(df_epoch_val['OS Month'], -df_epoch_val['Predicted'], df_epoch_val["Censored"])
            else:
                correct = (df_epoch_val['True Value'] == df_epoch_val['Predicted']).sum()
                accuracy = correct / len(df_epoch_val)
                # AUC Calculation
                if self.label_type == "classification":
                    if self.num_classes == 2 or self.num_classes == 1:
                        # Binary classification
                        y_true = df_epoch_val["True Value"].values
                        y_score = df_epoch_val["Probabilities"].values if "Probabilities" in df_epoch_val else None
                        if y_score is None and hasattr(self, 'train_probabilities'):
                            y_score = self.train_probabilities
                        auc_score = roc_auc_score(y_true, y_score)
                    else:
                        # Multi-class classification
                        y_true = df_epoch_val["True Value"].values
                        y_score = df_epoch_val[[str(i) for i in range(self.num_classes)]].values if all(str(i) in df_epoch_val.columns for i in range(self.num_classes)) else None
                        if y_score is not None:
                            auc_score = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
                        else:
                            auc_score = None
                epoch_val_score = auc_score
            
            # Calculate test score
            epoch_test_score = 0.0
            test_auc_score = None
            if self.parser_args.loss == "CoxPHLoss":
                epoch_test_score = concordance_index(df_epoch_test['OS Month'], -df_epoch_test['Predicted'], df_epoch_test["Censored"]) if self.parser_args.loss=="NegativeLogLikelihood" else concordance_index(df_epoch_test['OS Month'], -df_epoch_test['Predicted'], df_epoch_test["Censored"])
            else:
                correct_test = (df_epoch_test['True Value'] == df_epoch_test['Predicted']).sum()
                test_accuracy = correct_test / len(df_epoch_test)
                # Test AUC Calculation
                if self.label_type == "classification":
                    if self.num_classes == 2 or self.num_classes == 1:
                        # Binary classification
                        y_true_test = df_epoch_test["True Value"].values
                        y_score_test = df_epoch_test["Probabilities"].values if "Probabilities" in df_epoch_test else None
                        if y_score_test is not None:
                            test_auc_score = roc_auc_score(y_true_test, y_score_test)
                    else:
                        # Multi-class classification
                        y_true_test = df_epoch_test["True Value"].values
                        y_score_test = df_epoch_test[[str(i) for i in range(self.num_classes)]].values if all(str(i) in df_epoch_test.columns for i in range(self.num_classes)) else None
                        if y_score_test is not None:
                            test_auc_score = roc_auc_score(y_true_test, y_score_test, multi_class='ovr', average='macro')
                        else:
                            test_auc_score = None
                epoch_test_score = test_auc_score
            
            # Print per-epoch metrics
            if self.label_type == "classification":
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f} | Test Loss: {test_loss:.4f} | Val AUC: {epoch_val_score:.4f} | Test AUC: {epoch_test_score:.4f} | Val Acc: {accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
            else:
                print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {validation_loss:.4f} | Test Loss: {test_loss:.4f} | Val Score: {epoch_val_score:.4f} | Test Score: {epoch_test_score:.4f}")

            fold_dict["scheduler"].step(validation_loss)
            
            early_stopping(validation_loss, epoch_val_score, fold_dict["model"], vars(self.parser_args), id_file_name=self.setup_args.id, deg=fold_dict["deg"] if self.parser_args.model in ["PNAConv", "MMAConv", "GMNConv"] else None)
            if self.label_type == "classification":
                pbar.set_description(f"Train: {train_loss:.2f} Val: {validation_loss:.2f} Test: {test_loss:.2f} Val AUC: {epoch_val_score:.3f} Test AUC: {epoch_test_score:.3f} Best val: {early_stopping.best_eval_score:.3f} Patience: {early_stopping.counter}")
            else:
                pbar.set_description(f"Train: {train_loss:.2f} Val: {validation_loss:.2f} Test: {test_loss:.2f} Val Score: {epoch_val_score:.3f} Test Score: {epoch_test_score:.3f} Best val: {early_stopping.best_eval_score:.3f} Patience: {early_stopping.counter}")

            if early_stopping.early_stop or epoch == self.parser_args.epoch-1:
                # Save best metrics
                best_fold_loss = early_stopping.val_loss_min
                best_fold_score = early_stopping.best_eval_score
                best_fold_auc = auc_score if 'auc_score' in locals() else None
                
                # Store results for this split
                if self.label_type == "classification":
                    # Calculate final test metrics
                    correct_test = (df_epoch_test['True Value'] == df_epoch_test['Predicted']).sum()
                    test_accuracy = correct_test / len(df_epoch_test)
                    
                    # Calculate test AUC if applicable
                    test_auc = None
                    if self.num_classes == 2 or self.num_classes == 1:
                        y_true_test = df_epoch_test["True Value"].values
                        y_score_test = df_epoch_test["Probabilities"].values if "Probabilities" in df_epoch_test else None
                        if y_score_test is not None:
                            test_auc = roc_auc_score(y_true_test, y_score_test)
                    elif self.num_classes > 2:
                        y_true_test = df_epoch_test["True Value"].values
                        y_score_test = df_epoch_test[[str(i) for i in range(self.num_classes)]].values if all(str(i) in df_epoch_test.columns for i in range(self.num_classes)) else None
                        if y_score_test is not None:
                            test_auc = roc_auc_score(y_true_test, y_score_test, multi_class='ovr', average='macro')
                    
                    self.results.append([fold_dict['fold'], train_loss, validation_loss, test_loss, test_accuracy, test_auc, best_fold_auc])
                else:
                    # Regression case
                    if self.parser_args.loss == "CoxPHLoss":
                        test_ci = concordance_index(df_epoch_test['OS Month'], -df_epoch_test['Predicted'], df_epoch_test["Censored"])
                        self.results.append([fold_dict['fold'], train_loss, validation_loss, test_loss, test_ci])
                    else:
                        # Calculate regression metrics
                        test_r2 = r_squared_score(df_epoch_test['True Value'], df_epoch_test['Predicted'])
                        test_mse = mse(df_epoch_test['True Value'], df_epoch_test['Predicted'])
                        test_rmse = rmse(df_epoch_test['True Value'], df_epoch_test['Predicted'])
                        self.results.append([fold_dict['fold'], train_loss, validation_loss, test_loss, test_r2, test_mse, test_rmse])
                
                print("Best model lr:", fold_dict["optimizer"].param_groups[0]["lr"])
                self.parser_args.best_epoch = epoch
                fold_val_scores.append(early_stopping.best_eval_score)
                print("Early stopping the training...")
                break

        # Save results
        self.parser_args.ci_score = fold_val_scores[0] if fold_val_scores else 0.0
        self.parser_args.fold_ci_scores = fold_val_scores
        custom_tools.save_dict_as_json(vars(self.parser_args), self.setup_args.id, self.setup_args.MODEL_PATH)
        
        # Save model
        if not self.parser_args.fold:
            custom_tools.save_model(model=fold_dict["model"], fileName=self.setup_args.id, mode="SD", path=self.setup_args.MODEL_PATH)
            if self.parser_args.model == "PNAConv":
                custom_tools.save_pickle(fold_dict["deg"], f"{self.setup_args.id}_deg.pckl", self.setup_args.MODEL_PATH)
        
        # Save detailed results
        self.save_results()
        
        # Save test predictions
        if self.label_type == "regression":
            df_epoch_test.to_csv(os.path.join(self.setup_args.RESULT_PATH, f"{self.setup_args.id}_test_predictions.csv"), index=False)
        else:
            df_epoch_test.to_csv(os.path.join(self.setup_args.RESULT_PATH, f"{self.setup_args.id}_test_predictions.csv"), index=False)
        
        print(f"Train/Val/Test training completed!")
        print(f"Average validation score: {fold_val_scores[0]}")


    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 32 --dropout 0.0 --epoch 1000 --num_of_gcn_layers 2 --num_of_ff_layers 1 --gcn_h 128 --fcl 256 --en best_n_fold_17-11-2022 --weight_decay 0.0001 --factor 0.8 --patience 5 --min_lr 2e-05 --aggregators sum max --no-fold --label OSMonth --loss CoxPHLoss
    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 32 --dropout 0.0 --epoch 1000 --num_of_gcn_layers 2 --num_of_ff_layers 1 --gcn_h 128 --fcl 256 --en best_n_fold_17-11-2022 --weight_decay 0.0001 --factor 0.8 --patience 5 --min_lr 2e-05 --aggregators sum max --no-fold --label OSMonth --loss NegativeLogLikelihood
    # python train_test_controller.py --dataset_name JacksonFischer --model PNAConv --lr 0.001 --bs 32 --dropout 0.0 --epoch 1000 --num_of_gcn_layers 2 --num_of_ff_layers 1 --gcn_h 128 --fcl 256 --en best_n_fold_17-11-2022 --weight_decay 0.0001 --factor 0.8 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --label OSMonth --loss NegativeLogLikelihood        

    # full_training
    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 16 --dropout 0.0 --epoch 200 --num_of_gcn_layers 3 --num_of_ff_layers 2 --gcn_h 32 --fcl 512 --en best_full_training_week_15-12-2022 --weight_decay 0.0001 --factor 0.5 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --full_training --label OSMonth