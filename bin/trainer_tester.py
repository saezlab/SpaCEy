from cProfile import label
from sklearn.metrics import accuracy_score, f1_score, precision_score
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
            self.dataset = LungDataset(os.path.join(self.setup_args.S_PATH, f"../data/{self.parser_args.dataset_name}"), "Relapse")
            # RAW_DATA_PATH = os.path.join("../data", f"{dataset_name}/raw")
            #dataset = LungDataset(f"../data/{dataset_name}",  "Relapse")
        else:
            self.dataset = TissueDataset(os.path.join(self.setup_args.S_PATH, f"../data/{self.parser_args.dataset_name}", self.parser_args.unit),  self.parser_args.unit)
        # dataset = TissueDataset(os.path.join("../data/JacksonFischer/month"), "month")

        print("Number of samples:", len(self.dataset), self.parser_args.dataset_name,  self.parser_args.label)

        if self.parser_args.label == "OSMonth" or self.parser_args.loss == "CoxPHLoss":
            self.label_type = "regression"
            self.num_classes = 1

        elif self.parser_args.label == "treatment":
            self.label_type = "classification"
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()
            self.dataset.data.y = self.dataset.data.clinical_type
            self.unique_classes = set(self.dataset.data.clinical_type)
            self.num_classes = len(self.unique_classes)


        # WARN CURRENTLY DOESN'T PULL THE DATA NOT WORKING
        elif self.parser_args.label == "DiseaseStage":
            self.label_type = "classification"
            self.setup_args.criterion = torch.nn.CrossEntropyLoss()

        elif self.parser_args.label == "Relapse":
            print("Classification")
            self.label_type = "classification"
            # self.setup_args.criterion = torch.nn.CrossEntropyLoss()
            self.dataset.data.y = self.dataset.data.y
            self.unique_classes = torch.unique(self.dataset.data.y)
            # self.num_classes = len(self.unique_classes)
            self.num_classes = 1

        self.label_data = self.dataset.data.y
        if self.label_type=="classification":
            self.LT_ToIndex, self.LT_FromIndex = custom_tools.extract_LT(self.dataset.data.y)
            self.dataset.data.y = custom_tools.convert_wLT(self.dataset.data.y,self.LT_ToIndex)

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

        
        else:
            
            # self.samplers = custom_tools.k_fold_by_group(self.dataset)
            if self.parser_args.dataset_name == "Lung":
                self.samplers = custom_tools.k_fold_random_split(self.dataset)
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

            out = fold_dict["model"](data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.
            # print("Out shape", out.shape, torch.sigmoid(out), data.y)
            loss = None
            if self.parser_args.loss == "CoxPHLoss":#  or self.parser_args.loss == "NegativeLogLikelihood":
                loss = self.setup_args.criterion(out, data.y.to(self.device), data.is_censored.to(self.device))  # Compute the loss.    
            else:
                # print("Loss", data.y, out.squeeze(), out.shape)
                loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device)) # .to(self.device))  # Compute the loss.
                # print("Loss", loss, loss.item(), data.y.to(self.device), out.squeeze(), out)
            
            loss.backward()  # Derive gradients.
            
            pred_list.extend([val.item() for val in out.squeeze()])
            total_loss += float(loss.item())
            fold_dict["optimizer"].step()  # Update parameters based on gradients.
            fold_dict["optimizer"].zero_grad()  # Clear gradients

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
        pid_list, img_list, pred_list, true_list, tumor_grade_list, clinical_type_list, osmonth_list, censorship_list, relapse_list = [], [], [], [], [], [], [], [], []
        
        for data in loader:  # Iterate in batches over the training/test dataset.
            if data.y.shape[0]>1:
                out = model(data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)).type(torch.DoubleTensor).to(self.device) # Perform a single forward pass.

                loss = None
                if self.parser_args.loss == "CoxPHLoss":#  or self.parser_args.loss=="NegativeLogLikelihood":
                    loss = self.setup_args.criterion(out, data.y.to(self.device), data.is_censored.to(self.device))  # Compute the loss.    
                else:
                    # print("Classification loss")
                    # print(out.squeeze(),  data.y.to(self.device))
                    loss = self.setup_args.criterion(out.squeeze(), data.y.to(self.device))  # Compute the loss.

                total_loss += float(loss.item())

                true_list.extend([round(val.item(),6) for val in data.y])
                # WARN Correct usage of "max" ?
                if self.parser_args.loss == "CoxPHLoss":# == "regression":
                    pred_list.extend([val.item() for val in out.squeeze()])
                else:
                    probs = torch.sigmoid(out)
                    preds = (probs >= 0.5).float()
                    # print("preds, data.y", preds.squeeze(), data.y)
                    correct = (np.array(preds.squeeze().cpu()) == np.array(data.y)).sum()
                    # accuracy = correct / data.y.size(0)
                    # print("Accuracy", accuracy)
                    pred_list.extend([val for val in preds])
                
                #pred_list.extend([val.item() for val in data.y])
                # print(self.parser_args.dataset_name)
                if return_pred_df and self.parser_args.loss == "CoxPHLoss":
                    tumor_grade_list.extend([val for val in data.tumor_grade])
                    clinical_type_list.extend([val for val in data.clinical_type])
                    osmonth_list.extend([val.item() for val in data.osmonth])
                    censorship_list.extend([val.item() for val in data.is_censored])
                    pid_list.extend([val for val in data.p_id])
                    img_list.extend([val for val in data.img_id])
                    
                elif return_pred_df:

                    tumor_grade_list.extend([val for val in data.tumor_grade])
                    clinical_type_list.extend([val for val in data.clinical_type])
                    osmonth_list.extend([val.item() for val in data.osmonth])
                    relapse_list.extend([val.item() for val in data.y])
                    pid_list.extend([val for val in data.p_id])
                    img_list.extend([val for val in data.img_id])
                    
                    
                
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
                df = pd.DataFrame(list(zip(pid_list, img_list, true_list, pred_list, tumor_grade_list, clinical_type_list, osmonth_list,  label_list)),
                columns =["Patient ID","Image Number", 'True Value', 'Predicted', "Tumor Grade", "Clinical Type", "OS Month", "Fold#-Set"])
            
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

        # print(self.fold_dicts)
        for fold_dict in self.fold_dicts:
            # print(fold_dict["model"])
            best_val_loss = np.inf
            early_stopping = EarlyStopping(patience=self.parser_args.patience*2, verbose=True, model_path=self.setup_args.MODEL_PATH)

            print(f"########## Fold :  {fold_dict['fold']} ########## ")
            for epoch in (pbar := tqdm(range(self.parser_args.epoch), disable=False)):

                self.train(fold_dict)
                train_loss = self.test(fold_dict["model"], fold_dict["train_loader"],  fold_dict["fold"])
                validation_loss, df_epoch_val = self.test(fold_dict["model"], fold_dict["validation_loader"],  fold_dict["fold"], "validation", self.setup_args.plot_result)
                # test_loss, df_epoch_test = self.test(fold_dict["model"], fold_dict["test_loader"],  fold_dict["fold"], "test", self.setup_args.plot_result)
                epoch_val_score = 0.0   
                if self.parser_args.loss == "CoxPHLoss":
                    epoch_val_score = concordance_index(df_epoch_val['OS Month'], -df_epoch_val['Predicted'], df_epoch_val["Censored"]) if self.parser_args.loss=="NegativeLogLikelihood" else concordance_index(df_epoch_val['OS Month'], -df_epoch_val['Predicted'], df_epoch_val["Censored"])
                else:
                    
                    correct = (df_epoch_val['True Value'] == df_epoch_val['Predicted']).sum()
                    # print((df_epoch_val['True Value'][:10] == df_epoch_val['Predicted'][:10]))
                    # print(df_epoch_val['True Value'][:10], df_epoch_val['Predicted'][:10])
                    # print("correct, len(df_epoch_val)", correct, len(df_epoch_val))
                    # print(df_epoch_val['True Value'])
                    # print(df_epoch_val['Predicted'])
                    epoch_val_score = correct / len(df_epoch_val)
                    # print("Accuracy", accuracy)

            
                fold_dict["scheduler"].step(validation_loss)
                
                early_stopping(validation_loss, epoch_val_score, fold_dict["model"], vars(self.parser_args), id_file_name=self.setup_args.id, deg=self.fold_dicts[0]["deg"] if self.parser_args.model in ["PNAConv", "MMAConv", "GMNConv"] else None)
                # pbar.set_description(f"Train loss: {train_loss:.2f} Val. loss: {validation_loss:.2f} Val c_index: {epoch_val_ci_score} Patience: {early_stopping.counter}")
                pbar.set_description(f"Train loss: {train_loss:.2f} Val. loss: {validation_loss:.2f} Best val. score: {early_stopping.best_eval_score} Patience: {early_stopping.counter}")

                if early_stopping.early_stop or epoch==self.parser_args.epoch-1:
                    print("Best model lr:", fold_dict["optimizer"].param_groups[0]["lr"])
                    self.parser_args.best_epoch = epoch
                    fold_val_scores.append(early_stopping.best_eval_score)
                    print("Early stopping the training...")
                    break

        average_val_scores = sum(fold_val_scores)/len(fold_val_scores)
        print(f"Average validation score: {sum(fold_val_scores)/len(fold_val_scores)}")
        if average_val_scores > 0.65:
            self.parser_args.ci_score = average_val_scores
            self.parser_args.fold_ci_scores = fold_val_scores
            custom_tools.save_dict_as_json(vars(self.parser_args), self.setup_args.id, self.setup_args.MODEL_PATH)
            # print(f"Average validation score: {sum(fold_val_scores)/len(fold_val_scores)}")
            box_plt = sns.boxplot(data=fold_val_scores)
            fig = box_plt.get_figure()
            fig.savefig(os.path.join(self.setup_args.PLOT_PATH, f"{self.setup_args.id}.png"))
    
    
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


    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 32 --dropout 0.0 --epoch 1000 --num_of_gcn_layers 2 --num_of_ff_layers 1 --gcn_h 128 --fcl 256 --en best_n_fold_17-11-2022 --weight_decay 0.0001 --factor 0.8 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --label OSMonth --loss CoxPHLoss
    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 32 --dropout 0.0 --epoch 1000 --num_of_gcn_layers 2 --num_of_ff_layers 1 --gcn_h 128 --fcl 256 --en best_n_fold_17-11-2022 --weight_decay 0.0001 --factor 0.8 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --label OSMonth --loss NegativeLogLikelihood
    # python train_test_controller.py --dataset_name JacksonFischer --model PNAConv --lr 0.001 --bs 32 --dropout 0.0 --epoch 1000 --num_of_gcn_layers 2 --num_of_ff_layers 1 --gcn_h 128 --fcl 256 --en best_n_fold_17-11-2022 --weight_decay 0.0001 --factor 0.8 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --label OSMonth --loss NegativeLogLikelihood        

    # full_training
    # python train_test_controller.py --model PNAConv --lr 0.001 --bs 16 --dropout 0.0 --epoch 200 --num_of_gcn_layers 3 --num_of_ff_layers 2 --gcn_h 32 --fcl 512 --en best_full_training_week_15-12-2022 --weight_decay 0.0001 --factor 0.5 --patience 5 --min_lr 2e-05 --aggregators sum max --scalers amplification --no-fold --full_training --label OSMonth