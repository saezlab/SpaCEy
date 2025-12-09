import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from custom_tools import create_hyperparameter_combinations
from baseline_hyperparams import FastSurvivalSVM_params, RandomSurvivalForest_params, GradientBoostingSurvivalAnalysis_params
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.column import encode_categorical
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastSurvivalSVM
import sksurv.util as su
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis


dataset_name = "JacksonFischer"
dataset_name= "METABRIC"
random_state = 42
dataset_path = os.path.join("./data", dataset_name)
num_of_markers = 37 if dataset_name=="METABRIC" else 33
def get_dataset_df(agg="mean"):
    wanted_label = "OSmonth"
    unit = "month"
    count = 0
    columns = []
    data_list = []
    for fl in os.listdir(os.path.join(dataset_path, "raw")):
        if fl.endswith("features.pickle"):
            # print(fl)
            img, pid = fl.split("_")[:2]
            with open(os.path.join(dataset_path, "raw", f'{img}_{pid}_clinical_info.pickle'), 'rb') as handle:
                clinical_info_dict = pickle.load(handle)
            if pd.notna(clinical_info_dict["OSmonth"]):
                censored = None

                if "Patientstatus" in clinical_info_dict.keys():
                    censored = 0 if clinical_info_dict["Patientstatus"].lower().startswith("death") else 1
                elif "Overall Survival Status" in clinical_info_dict.keys():
                    censored = 0 if "deceased" in clinical_info_dict["Overall Survival Status"].lower() else 1

                with open(os.path.join(dataset_path, "raw", f'{img}_{pid}_features.pickle'), 'rb') as handle:
                    feature_arr = pickle.load(handle)
                    feature_arr = np.array(feature_arr)
                    if agg=="mean":
                        feature_arr = feature_arr.mean(axis=0)
                    elif agg=="sum":
                        feature_arr = feature_arr.sum(axis=0)
                    elif agg=="min":
                        feature_arr = feature_arr.min(axis=0)
                    elif agg=="max":
                        feature_arr = feature_arr.max(axis=0)

                row = feature_arr.copy()
                y_val = clinical_info_dict["OSmonth"]

                lst_clinical_info = clinical_info_dict.keys()
                lst_clinical_info = sorted(lst_clinical_info)

                if len(columns)==0:
                    for i in range(len(feature_arr)):
                        columns.append(f"Marker_{i}")
                    for ci in lst_clinical_info:
                        columns.append(ci)
                    columns.append("img_id")
                    columns.append("p_id")



                for key in lst_clinical_info:
                    row = np.append(row, clinical_info_dict[key])

                row = np.append(row, img)
                row = np.append(row, pid)

                data_list.append(row)
                # print(feature_arr)
                count+=1

    df_features = pd.DataFrame(data_list, columns=columns)
    return df_features


def load_json(file_path):
    """Loads the json file for given path

    Args:
        file_path (string): file path

    Returns:
        dict: dict of the json
    """
    
    with open(file_path, 'r') as fp:
        l_dict = json.load(fp)
    return l_dict

import json
json_fl = load_json(f"./data/{dataset_name}/folds.json")


aggregator = ["sum", "mean", "min", "max"]
estimators = ["FastSurvivalSVM", "RandomSurvivalForest", "GradientBoostingSurvivalAnalysis"]

hyper_param_combs_dict = {"FastSurvivalSVM":create_hyperparameter_combinations(FastSurvivalSVM_params), "RandomSurvivalForest":create_hyperparameter_combinations(RandomSurvivalForest_params), "GradientBoostingSurvivalAnalysis":create_hyperparameter_combinations(GradientBoostingSurvivalAnalysis_params)}

result_df_cols  = []
all_results = []
for agg in aggregator:
    df_dataset = get_dataset_df(agg=agg)
    all_data_y =[]
    all_status = []
    for ind, row in df_dataset.iterrows():
        if dataset_name == "METABRIC":
            patient_status = True if row["diseasestatus"]!="Living" else False
        else:
            patient_status = True if row["Patientstatus"].lower().startswith("death") else False
        all_data_y.append((patient_status, float(row["OSmonth"])))
        all_status.append(patient_status)

    df_dataset["data_y"] = all_data_y
    df_dataset["status"] = all_status
    # print(df_dataset)

    # List to save sampler triplet
    samplers = []

    for fold in json_fl["fold_img_id_dict"]:
        train_idx = list(df_dataset.loc[df_dataset.img_id.isin(json_fl["fold_img_id_dict"][fold][0]),:].index)
        test_idx = list(df_dataset.loc[df_dataset.img_id.isin(json_fl["fold_img_id_dict"][fold][1]),:].index)

        samplers.append((
                (fold), # fold number
                (train_idx),
                (test_idx)))
    
    for est_name in estimators:
        hyper_param_combs = hyper_param_combs_dict[est_name]
        for ind, comb in tqdm(enumerate(hyper_param_combs), total=len(hyper_param_combs), desc=f"{est_name}, {agg}"):
            result_df_cols.append(f"{est_name}-{ind}-{agg}")
            estimator = None
            if est_name == "FastSurvivalSVM":
                estimator = FastSurvivalSVM(**comb)
            elif est_name == "RandomSurvivalForest":
                estimator = RandomSurvivalForest(**comb)
            elif est_name == "GradientBoostingSurvivalAnalysis":
                estimator = GradientBoostingSurvivalAnalysis(**comb)
            else:
                raise Exception("Please enter a valid estimator...")
            
            k_fold_cindex = []
            for fold_id, train_idx, test_idx in samplers:
                train_df = df_dataset.iloc[train_idx]
                # print(train_df.columns)
                data_x = train_df.iloc[:,:num_of_markers]
                
                data_y = su.Surv.from_arrays(train_df["status"], train_df["OSmonth"].astype("float").add(0.1))

                test_df = df_dataset.iloc[test_idx]
                test_data_x = test_df.iloc[:,:num_of_markers]
                test_data_y = su.Surv.from_arrays(test_df["status"], test_df["OSmonth"].astype("float").add(0.1))
                estimator.fit(data_x, data_y)

                test_cindex = concordance_index_censored(
                test_df["status"],
                test_df["OSmonth"],
                estimator.predict(test_data_x))
                k_fold_cindex.append(test_cindex[0])
            all_results.append(k_fold_cindex)
            # print(f"{est_name}\t{ind}\t{agg}:", round(sum(k_fold_cindex)/len(k_fold_cindex),2))

print(len(all_results))
print(len(result_df_cols))
df_results = pd.DataFrame(np.array(all_results).T, columns=result_df_cols)

df_results.to_csv(f"./data/out_data/baseline_predictors/{dataset_name}_pseudobulk_results.csv")

