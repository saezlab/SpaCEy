import os
import json
import argparse
import numpy as np
import pandas as pd
from evaluation_metrics import r_squared_score, mse, rmse, mae
from eval import concordance_index

def calculate_reg_scores_from_preds(folder_path, file_name):
    experiment_name = os.path.split(folder_path)[-1]
    df_results = pd.read_csv(os.path.join(folder_path, file_name), sep=",")
    job_id = file_name.split(".csv")[0]
    
    result_list = [experiment_name, job_id]
    for idx, val in enumerate(["train", "validation", "test"]):

        df_tvt = df_results.loc[(df_results['Fold#-Set'].str[2:] == val)]
        col_name = "True Value"
        if "OS Month (log)" in df_tvt.columns:
            col_name = "OS Month (log)"

        r2_score = r_squared_score(df_tvt[col_name], df_tvt['Predicted'])
        mser = mse(df_tvt[col_name], df_tvt['Predicted'])
        result_list.extend([r2_score, mser])
    
    return result_list

def calculate_all_reg_scores(folder_path_list):


    header = ["experiment name", "job id", "train_r2 score", "train_mse", "train_mae", "val_r2 score", "val_mse", "val_mae", "test_r2 score", "val_mse", "test_mae"]
    all_results = []
    for folder_path in folder_path_list:
        
        for fl in os.listdir(folder_path):
            if fl.endswith(".csv"):
                try:

                
                    all_results.append(calculate_mae_scores_from_preds(folder_path, fl))
                except:
                    print(f"{fl} is not proper!")
                    pass
        
    df_results = pd.DataFrame (all_results, columns = header)
    # df_results.sort_values(by=["val_r2 score", "test_r2 score"], inplace=True, ascending=False)
    df_results.sort_values(by=["test_mae", "val_mae"], inplace=True, ascending=True)

    return df_results


def calculate_mae_scores_from_preds(folder_path, file_name):
    experiment_name = os.path.split(folder_path)[-1]
    df_results = pd.read_csv(os.path.join(folder_path, file_name), sep=",")
    job_id = file_name.split(".csv")[0]
    
    result_list = [experiment_name, job_id]
    tvt = True
    if tvt:
        for idx, val in enumerate(["train", "validation", "test"]):

            col_name = "True Value"
            if "OS Month (log)" in df_results.columns:
                col_name = "OS Month (log)"

            df_tvt = df_results.loc[(df_results['Fold#-Set'].str[2:] == val)]

            r2_score = r_squared_score(df_tvt[col_name], df_tvt['Predicted'])
            mser = mse(df_tvt[col_name], df_tvt['Predicted'])
            mean_abs_err = mae(df_tvt[col_name], df_tvt['Predicted'])
            result_list.extend([r2_score, mser, mean_abs_err])
    else:
        col_name = "True Value"
        if "OS Month (log)" in df_results.columns:
            col_name = "OS Month (log)"

        r2_score = r_squared_score(df_results[col_name], df_results['Predicted'])
        mser = mse(df_results[col_name], df_results['Predicted'])
        mean_abs_err = mae(np.exp(df_results[col_name]), np.exp(df_results['Predicted']))
        result_list.extend([r2_score, mser, mean_abs_err])
    
    return result_list


def calculate_cindex_scores_from_preds(folder_path, file_name):
    experiment_name = os.path.split(folder_path)[-1]
    df_results = pd.read_csv(os.path.join(folder_path, file_name), sep=",")
    job_id = file_name.split(".csv")[0]
    
    result_list = [experiment_name, job_id]
    tvt = True
    
    if tvt:
        for idx, val in enumerate(["train", "validation", "test"]):

            true_val = "True Value"
            censor_val = "Censored"

            df_tvt = df_results.loc[(df_results['Fold#-Set'].str[2:] == val)]

            c_index = concordance_index(df_tvt[true_val], -df_tvt['Predicted'], df_tvt[censor_val]) if "NegativeLogLikelihood" in folder_path else concordance_index(df_tvt[true_val], df_tvt['Predicted'], df_tvt[censor_val])
            
            result_list.extend([c_index])
    
    return result_list


def calculate_all_cindex_scores(folder_path_list):


    header = ["experiment name", "job id", "train_cindex", "val_cindex", "test_cindex"]
    all_results = []
    for folder_path in folder_path_list:
        
        for fl in os.listdir(folder_path):
            if fl.endswith(".csv"):
                try:

                
                    all_results.append(calculate_cindex_scores_from_preds(folder_path, fl))
                except:
                    print(f"{fl} is not proper!")
                    pass
        
    df_results = pd.DataFrame (all_results, columns = header)
    print(df_results)
    df_results.sort_values(by=["val_cindex", "test_cindex"], inplace=True, ascending=False)
    # df_results.sort_values(by=["test_cindex"], inplace=True, ascending=True)

    return df_results

def get_average_scores_from_json(folder_path_list):
    # returns JSON object as 
    # a dictionary
    header = ["experiment name", "job id", "c_index"]
    all_results = []

    for folder_path in folder_path_list:
        
        for fl in os.listdir(folder_path):
            if fl.endswith(".json"):
                job_id = fl.rsplit(".", maxsplit=1)[0]
                fl = open(os.path.join(folder_path, fl))
                data = json.load(fl)
                all_results.append([data["en"], job_id, data["ci_score"]])
            
    df_results = pd.DataFrame (all_results, columns = header)
    
    df_results.sort_values(by=["c_index"], inplace=True, ascending=False)
    print(df_results[:10])


# calculate_cindex_scores_from_preds("./results/idedFiles/GATV2_CoxPHLoss_month_24-11-2023", "4vzHDHJhBEhzvI91la51HQ.csv")
# "./results/idedFiles/GATV2_CoxPHLoss_month_24-11-2023",
# df= calculate_all_cindex_scores([ "./results/idedFiles/PNAConv_CoxPHLoss_month_24-11-2023"])
# df= calculate_all_cindex_scores(["./results/idedFiles/GATV2_NegativeLogLikelihood_fixed_dataset_13-12-2023"])
# df= calculate_all_cindex_scores(["./results/idedFiles/GATV2_NegativeLogLikelihood_fixed_dataset_13-12-2023"])
# df= calculate_all_cindex_scores(["./results/idedFiles/PNAConv_NegativeLogLikelihood_month_30-11-2023", "./results/idedFiles/GATV2_NegativeLogLikelihood_month_04-12-2023"])
# df= calculate_all_cindex_scores(["./results/idedFiles/GATV2_NegativeLogLikelihood_month_04-12-2023# "])
# ./results/idedFiles/METABRIC_PNAConv_NegativeLogLikelihood_fixed_dataset_04-01-2024
# df= calculate_all_cindex_scores(["./results/idedFiles/METABRIC_PNAConv_NegativeLogLikelihood_fixed_dataset_04-01-2024"])
# print(df)
# print(calculate_all_reg_scores(["./data/out_data/PNA_training_year_30-11-2022"]))
# print(calculate_all_reg_scores(["./data/out_data/PNA_training_week_07-12-2022"]))
# print(calculate_all_reg_scores(["./data/out_data/best_n_fold_week_14-12-2022"]))

# print(calculate_all_reg_scores(["./data/out_data/GAT_training_month_nolog_21-12-2022", "./data/out_data/PNA_training_month_nolog_15-12-2022"]))
# print(calculate_all_reg_scores(["./bin/jobs/PNAConv_os_nolog_large_02-02-2023"]))

# print(calculate_mae_scores_from_preds("./data/out_data/best_full_training_week_15-12-2022", "CETuHyGFgP3ZZ3WwgqlUeA.csv"))


# print(calculate_mae_scores_from_preds("./data/out_data/PNAConv_PNA_MSE_week_lognorm_30-06-2023", "Vs2a-oNH7FLupUB80hJY9w.csv"))

# print(calculate_all_reg_scores(["./data/out_data/PNAConv_os_nolog_large_6-26-2023_h_loss", "./data/out_data/PNAConv_PNA_Huber_month_30-06-2023", "./data/out_data/PNAConv_PNA_MSE_month_30-06-2023", "./data/out_data/PNAConv_PNA_MSE_week_lognorm_30-06-2023"]))


# print(calculate_all_reg_scores(["./data/out_data/PNAConv_PNA_Huber_month_30-06-2023", "./data/out_data/PNAConv_PNA_MSE_week_lognorm_30-06-2023", "./data/out_data/PNAConv_PNA_MSE_month_30-06-2023"]))

# print(calculate_all_reg_scores([]))

# print(calculate_all_reg_scores(["./models/METABRIC_GATV2_CoxPHLoss_10_fold_gpusaez_14-04-2024"]))
# get_average_scores_from_json(["./models/METABRIC_GATV2_CoxPHLoss_10_fold_gpusaez_14-04-2024"])
# get_average_scores_from_json(["../models/METABRIC_GATV2_CoxPHLoss_10_fold_gpu_14-04-2024", "../models/METABRIC_GATV2_CoxPHLoss_10_fold_gpusaez_14-04-2024"])