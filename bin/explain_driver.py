import joblib
import torch
import sys
import numpy as np
import custom_tools
import os
import plotting
import argparse
from dataset import TissueDataset, LungDataset
from explainer_cli_classification import Custom_Explainer
import pytorch_lightning as pl
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(description='Explainer arguments...')
    
parser.add_argument(
    '--dataset_name',
    type=str,
    default="JacksonFischer",
    metavar='ds',
    help='dataset name (default: JacksonFischer, alternative METABRIC)')

parser.add_argument(
    '--exp_name',
    type=str,
    default="PNAConv",
    metavar='mn',
    help='model name (default: PNAConv)')

parser.add_argument(
    '--dataset_indices',
    type=str,   # comma separated list of indices
    default=None,
    metavar='di',
    help='dataset indices (default: None)')

parser.add_argument(
    '--job_id',
    type=str,
    metavar='ji',
    help='job id')

args = parser.parse_args()
print(args.exp_name, args.dataset_name)
S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
# dataset_name = "JacksonFischer"
# dataset_name = args.dataset_name
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", f"{args.dataset_name}/raw")


"""
job_id = "sE3-V1wnbN4AgiUbKeP5oA"
args  = custom_tools.load_json(f"../models/best_full_training_22-11-2022/{job_id}.json")
deg = custom_tools.load_pickle(f"../models/best_full_training_22-11-2022/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = "../models/best_full_training_22-11-2022", model_type = "SD", args = args, deg=deg)
"""


"""job_id = "VojrEHdFRPUTsJ3zGTiPcQ"
args  = custom_tools.load_json(f"../models/PNAConv_PNA_MSE_month_30-06-2023/{job_id}.json")
args["num_node_features"] = 33
deg = custom_tools.load_pickle(f"../models/PNAConv_PNA_MSE_month_30-06-2023/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = "../models/PNAConv_PNA_MSE_month_30-06-2023", model_type = "SD", args = args, deg=deg)"""


"""# Read json file
job_id = "5rBzqSUcY3fkBsmCtnU3Mw"
device =  custom_tools.get_device()
args  = custom_tools.load_json(f"../models/PNAConv_CoxPHLoss_month_24-11-2023/{job_id}.json")
# args["num_node_features"] = 33
deg = custom_tools.load_pickle(f"../models/PNAConv_CoxPHLoss_month_24-11-2023/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = "../models/PNAConv_CoxPHLoss_month_24-11-2023", model_type = "SD", args = args, deg=deg, device=device)"""


# Read json file
# exp_name = "GATV2_NegativeLogLikelihood_month_04-12-2023"
# job_id = "YyroGgMa_H4xn_ctP3C5Zw"
# exp_name = "METABRIC_PNAConv_NegativeLogLikelihood_fixed_dataset_04-01-2024"
# job_id = "dGMbZiU1cccDyS_Udt2xGg"

# exp_name = args.exp_name
# job_id = args.job_id

"""device =  custom_tools.get_device()
model_params  = custom_tools.load_json(f"../models/{args.exp_name}/{args.job_id}.json")
# args["num_node_features"] = 33
deg = None
if "PNA" in args.exp_name:
    deg = custom_tools.load_pickle(f"../models/{args.exp_name}/{args.job_id}_deg.pckl")
model = custom_tools.load_model(f"{args.job_id}_SD", path = f"../models/{args.exp_name}", model_type = "SD", args = model_params, deg=deg, device=device)


dataset = TissueDataset(os.path.join(f"../data/{args.dataset_name}", "month"),  "month")

Path( os.path.join("../plots/explanations", args.dataset_name, f"{args.exp_name}_{args.job_id}") ).mkdir(parents=True, exist_ok=True)
Path(f"../data/out_data/adatafiles/{args.dataset_name}" ).mkdir(parents=True, exist_ok=True)"""

device =  custom_tools.get_device()
model_params  = custom_tools.load_json(f"../models/{args.exp_name}/{args.job_id}.json")
# args["num_node_features"] = 33
deg = None
if "PNA" in args.exp_name:
    deg = custom_tools.load_pickle(f"../models/{args.exp_name}/{args.job_id}_deg.pckl")
model = custom_tools.load_model(f"{args.job_id}_SD", path = f"../models/{args.exp_name}", model_type = "SD", args = model_params, deg=deg, device=device)


dataset = LungDataset(os.path.join(f"../data/{args.dataset_name}"),  "Progression")
# print(len(dataset))
Path( os.path.join("../plots/explanations", args.dataset_name, f"{args.exp_name}_{args.job_id}") ).mkdir(parents=True, exist_ok=True)
Path(f"../data/out_data/adatafiles/{args.dataset_name}" ).mkdir(parents=True, exist_ok=True)

if args.dataset_indices is not None:
    dataset_indices = [int(idx) for idx in args.dataset_indices.split(",")]
    print(dataset_indices)
    dataset = dataset[dataset_indices]  
else:
    dataset = dataset.shuffle()


random_seed = 42 # , 21, 1, 12, 123, 1234, 2, 23, 234, 2345]
all_edges_list = []
custom_tools.set_seeds(seed=42, deterministic=True)
explainer = Custom_Explainer(model, args.dataset_name, dataset, args.exp_name, args.job_id, seed=random_seed)


for lr in [0.1]:# , 0.01, 0.001, 0.0001]:
    edge_idx = explainer.explain(epoch=100, lr=lr)



# python explain_driver.py --dataset_name JacksonFischer --exp_name "GATV2_NegativeLogLikelihood_month_04-12-2023" --job_id YyroGgMa_H4xn_ctP3C5Zw

# python explain_driver.py --dataset_name JacksonFischer --exp_name "JacksonFischer_Final" --job_id Q35QNYtJGirq4yCX3KzaKQ

# python explain_driver.py --dataset_name JacksonFischer --exp_name "JacksonFischer" --job_id Wwz8zqJAi6uciz73dC9vVw
# python explain_driver.py --dataset_name JacksonFischer --exp_name "JacksonFischer" --job_id 2mYvWJwUarIJkw0vvUbl3Q
# python explain_driver.py --dataset_name METABRIC --exp_name "METABRIC" --job_id pp_zypM0misNEo00MY-0nA
# python explain_driver.py --dataset_name METABRIC --exp_name "METABRIC" --job_id 9U43MVNB9CQHuwnTN6t0kg
# python explain_driver.py --dataset_name Lung --exp_name "Lung" --job_id Jlj4RO45BJ5KGxuEpOMiuw
# python explain_driver.py --dataset_name Lung --exp_name "Lung" --job_id TomhmA2qwAXrWz6Y697Dkw
# python explain_driver.py --dataset_name Lung --exp_name "Lung" --job_id Lung
