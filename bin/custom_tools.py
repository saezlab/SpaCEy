import os
import torch
import json
import pickle
import shutil
import random
import secrets
import argparse
import numpy as np
import pandas as pd
import pathlib
import scanpy as sc
import networkx as nx
from pathlib import Path
from model import CustomGCN
import pytorch_lightning as pl
from torch_geometric import utils
from data_preparation import get_basel_zurich_staining_panel
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold, StratifiedKFold

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")

# K-Fold cross validation index creator function
# Dataset idices and ratios must be supplied
# Return triplet of samplers for amount of wanted fold
def k_fold_ttv(dataset, dataset_name, T2VT_ratio,V2T_ratio, shuffle_VT = False, split_tvt_by_group = True):
    """Splits dataset into Train, Validation and Test sets

    Args:
        dataset (_type_): Data to be splitted
        T2VT_ratio (int): Train / (Valid + Test)
        V2T_ratio (int): Valid / Test
    """



    # List to save sampler triplet
    samplers = []

    if split_tvt_by_group:
        fold_count = 1
        # train_idx, valid_idx, test_idx = split_by_group(dataset)
        train_idx, valid_idx, test_idx = get_train_val_test_split_by_pid(dataset, dataset_name)
        # print("set(train_idx)&set(valid_idx)", set(train_idx)&set(valid_idx))
        # print("set(train_idx)&set(test_idx)", set(train_idx)&set(test_idx))

        samplers.append((
                (fold_count),
                (torch.utils.data.SubsetRandomSampler(train_idx)),
                (torch.utils.data.SubsetRandomSampler(valid_idx)),
                (torch.utils.data.SubsetRandomSampler(test_idx))))
        
        return samplers

    fold_T2VT=KFold(n_splits=T2VT_ratio+1)
    fold_V2T =KFold(n_splits=V2T_ratio+1)
    # Fold count init
    fold_count = 0

    # Pulling indexes for sets
    for (train_idx, valid_test_idx) in fold_T2VT.split(dataset):
        for (valid_idx, test_idx) in fold_V2T.split(valid_test_idx):

            fold_count += 1

            valid_idx = valid_test_idx[valid_idx]
            test_idx = valid_test_idx[test_idx]


            samplers.append((
                (fold_count),
                (torch.utils.data.SubsetRandomSampler(train_idx)),
                (torch.utils.data.SubsetRandomSampler(valid_idx)),
                (torch.utils.data.SubsetRandomSampler(test_idx))))

            if not shuffle_VT:
                break

    return samplers

def k_fold_by_group(dataset, n_of_folds=10, group_name="p_id"):
    """Splits dataset into Train, Validation and Test sets

    Args:
        dataset (_type_): Data to be splitted
        T2VT_ratio (int): Train / (Valid + Test)
        V2T_ratio (int): Valid / Test
    """

    lst_groups = []
    for item in dataset:
        lst_groups.append([item.p_id, item.img_id, item.clinical_type, item.tumor_grade, item.osmonth])

    df_dataset = pd.DataFrame(lst_groups, columns=["p_id", "img_id", "clinical_type", "tumor_grade", "osmonth"])

    group_kfold = GroupKFold(n_splits=n_of_folds)
    
    # tumor grade is just a random column
    group_kfold.get_n_splits(X=df_dataset["tumor_grade"], groups = df_dataset[group_name])
    
    
    # List to save sampler triplet
    samplers = []

    for i, (train_idx, test_idx) in enumerate(group_kfold.split(X = df_dataset["tumor_grade"], groups = df_dataset[group_name])):
        
        samplers.append((
            (i), # fold number
            (torch.utils.data.SubsetRandomSampler(train_idx)),
            (torch.utils.data.SubsetRandomSampler(test_idx))))

    return samplers


def k_fold_random_split(dataset, n_of_folds=10, seed=42):
    """Splits dataset into Train and Test sets using random K-Fold (no grouping)

    Args:
        dataset (_type_): Data to be split
        n_of_folds (int): Number of folds
        seed (int): Random seed for reproducibility
    """

    # Extract relevant data (or dummy columns if not needed)
    lst_data = []
    for item in dataset:
        lst_data.append([item.p_id, item.img_id, item.clinical_type, item.tumor_grade, item.osmonth])
    
    df_dataset = pd.DataFrame(lst_data, columns=["p_id", "img_id", "clinical_type", "tumor_grade", "osmonth"])
    
    kf = KFold(n_splits=n_of_folds, shuffle=True, random_state=seed)
    
    samplers = []
    for i, (train_idx, test_idx) in enumerate(kf.split(df_dataset)):
        samplers.append((
            i,  # fold number
            torch.utils.data.SubsetRandomSampler(train_idx),
            torch.utils.data.SubsetRandomSampler(test_idx)
        ))

    return samplers

def save_model(model: CustomGCN,fileName ,mode: str, path = os.path.join(os.curdir, "..", "models")):
    """
    Saves the pytoch model, has 3 modes.
    Models have extension ".mdl", 
    Mode SD: Just save the state dict., user needs to hold hyperparameters
    Mode SDH: State dict and hyperparameters are saved, in DIFFERENT files
    Mode EM: Entire model is saved
    For now only works with "model_dgermen", NOT generic

    Args:
        model (CustomGCN): Model to be saved
        fileName (str): Model's saved file name.
        mode (str): Mode to save the model
        path (_type_, optional): path that model will be saved to. Defaults to os.curdir().
    """
    path_model = os.path.join(path, fileName + "_" + mode + ".mdl") 
    path_hyp = os.path.join(path + fileName + "_" + mode  + ".hyp")

    if mode == "SD":
        torch.save(model.state_dict(), path_model)

    elif mode == "SDH": 
        torch.save(model.state_dict(), path_model)
        
        try: 
            model_hyp = {
                "type" : model.model_type,
                "args" : model.pars
            }

        except KeyError:
            raise KeyError("Model doesn't have type attribute, add manually or use updated 'model_dgermen'.")

        pickle_out = open(path_hyp,"wb")
        pickle.dump(model_hyp, pickle_out)
        pickle_out.close()

    elif mode == "EM":
        torch.save(model, path_model)

    print(f"Model saved with session id: {fileName}!")
    print(f"Saved model path: {path_model}!")


def load_model(fileName: str, path =  os.curdir, model_type: str = "NONE", args: dict = {}, deg = None, gpu_id=None, label_type = "regression", device=None):
    """Models the specified model and returns it

    Args:
        fileName (str): Name of the file without the extension
        path (str, optional): Path to the directory of the saved model. Defaults to os.curdir.
        model_type (str, optional): If saved with SD mode, need to supply model type. Defaults to "NONE".
        args (dict, optional): If saved with SD mode, need to supply model creation hyperparameter. Defaults to {}.

    Raises:
        ValueError: 
        ValueError: 

    Returns:
        CustomGCN: Ready model
    """
    # print(args)
    path_model = os.path.join(path, fileName + ".mdl")
    path_hyp = os.path.join(path, fileName + ".hyp")

    mode = fileName.rsplit("_",1)[-1]
    
    if mode == "SD":
        if model_type == "NONE":
            raise ValueError("For SD mode, model_type must be properly defined!")
        if args == {}:
            raise ValueError("For SD mode, args must be supplied properly!")

        """model = CustomGCN(
                    type = args["model"],
                    num_node_features = args["num_node_features"], 
                    num_gcn_layers= args["num_of_gcn_layers"], 
                    num_ff_layers=args["num_of_ff_layers"], 
                    gcn_hidden_neurons=args["gcn_h"], 
                    ff_hidden_neurons=args["fcl"], 
                    dropout=args["dropout"],
                    aggregators=args["aggregators"],
                    scalers=args["scalers"],
                    deg = deg
                        )"""
        model = CustomGCN(
                    type = args["model"],
                    num_node_features = args["num_node_features"], 
                    num_gcn_layers=args["num_of_gcn_layers"], 
                    num_ff_layers=args["num_of_ff_layers"], 
                    gcn_hidden_neurons=args["gcn_h"], 
                    ff_hidden_neurons=args["fcl"], 
                    dropout=args["dropout"],
                    aggregators=args["aggregators"],
                    scalers=args["scalers"],
                    deg = deg, # Comes from data not hyperparameter
                    # num_classes = args["num_classes"],
                    heads = args["heads"],
                    label_type = label_type# args["label"]
                        )

        model.load_state_dict(torch.load(path_model, map_location=device))
        model.eval()

        return model

    elif mode == "SDH":

        pickle_in = open(path_hyp,"rb")
        hyp = pickle.load(pickle_in)
        pickle_in.close()

        model_type = hyp["type"]
        args = hyp["args"]

        model = CustomGCN(
                    type = model_type,
                    num_node_features = args["num_node_features"], 
                    num_gcn_layers=args["num_gcn_layers"], 
                    num_ff_layers=args["num_ff_layers"], 
                    gcn_hidden_neurons=args["gcn_hidden_neurons"], 
                    ff_hidden_neurons=args["ff_hidden_neurons"], 
                    dropout=args["dropout"],
                    aggregators=args["aggregators"],
                    scalers=args["scalers"],
                    deg = args["deg"] 
                        )
        # print(model)
        model.load_state_dict(torch.load(path_model))
        model.eval()

        return model

    
    elif mode == "EM":

        model = torch.load(path_model)
        model.eval()

        return model


def get_device(gpu_id=None):
    """Returns the available device, default priority is "cuda"

    Returns:
        string: cuda or cpu
    """

    use_gpu = torch.cuda.is_available()

    device = "cpu"
    
    if use_gpu:
        device = "cuda"
        if gpu_id:
            device=f"cuda:{gpu_id}"
        print("GPU is available on this device!")
    else:
        print("CPU is available on this device!")

    return device


def save_dict_as_json(s_dict, file_name, path):
    """
    Save dictionary as a json file
    """

    with open(os.path.join(path,file_name+".json"), "w") as write_file:
        json.dump(s_dict, write_file, indent=4)


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


def save_pickle(obj, file_name: str, path =  os.curdir):

    pickle_out = open(os.path.join(path, file_name),"wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def load_pickle(path_obj):
    pickle_in = open(path_obj,"rb")
    obj = pickle.load(pickle_in)
    pickle_in.close()
    return obj


def generate_session_id():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16)  


def general_parser() -> argparse.Namespace:
    """Used inside a file, makes the file able to parse CLI arguments.

    Returns:
        argparse.Namespace: argparse namespace that includes supplied argument values
    """

    parser = argparse.ArgumentParser(description='GNN Arguments')
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="JacksonFischer",
        metavar='ds',
        help='dataset name (default: JacksonFischer, alternatives METABRIC, Lung)')
    
    parser.add_argument(
        '--model',
        type=str,
        default="PNAConv",
        metavar='mn',
        help='model name (default: PNAConv)')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.001)')

    parser.add_argument(
        # '--batch-size',
        '--bs',
        type=int,
        default=32,
        metavar='BS',
        help='batch size (default: 32)')

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.20, # 0.1 0.2 0.3 0.5
        metavar='DO',
        help='dropout rate (default: 0.20)')

    parser.add_argument(
        '--epoch',
        type=int,
        default=5,
        metavar='EPC',
        help='Number of epochs (default: 50)')

    parser.add_argument(
        '--num_of_gcn_layers',
        type=int,
        default=2,
        metavar='NGCNL',
        help='Number of GCN layers (default: 2)')

    parser.add_argument(
        '--num_of_ff_layers',
        type=int,
        default=2,
        metavar='NFFL',
        help='Number of FF layers (default: 2)')
        
    parser.add_argument(
        '--gcn_h',
        type=int,
        default=128,
        metavar='GCNH',
        help='GCN hidden channel (default: 128)')

    parser.add_argument(
        '--fcl',
        type=int,
        default=128,
        metavar='FCL',
        help='Number of neurons in fully-connected layer (default: 128)')

    parser.add_argument(
        '--en',
        type=str,
        default="test_experiment",
        metavar='EN',
        help='the name of the experiment (default: test_experiment)')

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.001,
        metavar='WD',
        help='weight decay (default: 0.001)')

    parser.add_argument(
        '--factor', # 0.5 0.8, 0.2
        type=float,
        default=0.5,
        metavar='FACTOR',
        help='learning rate reduce factor (default: 0.5)')

    parser.add_argument(
        '--patience', # 5, 10, 20
        type=int,
        default=5,
        metavar='PA',
        help='patience for learning rate scheduling (default: 5)')

    parser.add_argument(
        '--min_lr',
        type=float,
        default=0.00002,#0.0001
        metavar='MLR',
        help='minimum learning rate (default: 0.00002)')

    parser.add_argument(
        '--aggregators',
        # WARN This use of "+" doesn't go well with positional arguments
        nargs='+',
        # PROBLEM How to feed a list in CLI? Need to edit generator?
        # Take string split
        type = str,
        # ARBTR Change to something meaningful
        default= ["sum","mean"], # "sum", "mean", "min", "max", "var" and "std".
        metavar='AGR',
        help= "aggregator list for PNAConv")

    parser.add_argument(
        '--scalers',
        # WARN This use of "+" doesn't go well with positional arguments
        nargs='+',
        # PROBLEM How to feed a list in CLI? Need to edit generator?
        type= str,
        default= ["amplification","identity"], # "identity", "amplification", "attenuation", "linear" and "inverse_linear"
        metavar='SCL',
        help='Set of scaling function identifiers,')

    parser.add_argument(
        '--heads',
        type = int,
        default=1,
        metavar="NOH",
        help='number of heads for GATConv'
    )

    parser.add_argument(
        '--fold',
        type= bool,
        action=argparse.BooleanOptionalAction,
        default= True,
        metavar='F',
        help='Perform train/val/test or n-fold x-val (--fold, --no-fold),')

    parser.add_argument(
        '--full_training',
        type= bool,
        action=argparse.BooleanOptionalAction,
        default= False,
        metavar='F',
        help='Perform full_training, default: --no-full_training (--full_training, --no-full_training),')
    
    parser.add_argument(
        '--t_v_t',
        type= bool,
        action=argparse.BooleanOptionalAction,
        default= False,
        metavar='F',
        help='Perform t_v_t, default: --no-t_v_t (--t_v_t, --no-t_v_t),')

    parser.add_argument(
        '--label',
        type= str,
        default= "OSMonth", # "identity", "amplification", "attenuation", "linear" and "inverse_linear"
        metavar='LBL',
        help='Which property of the dataset will be used as label')
    
    parser.add_argument(
        '--loss',
        type= str,
        default= "MSE", # "MSE", "HuberLoss"
        metavar='LOSS',
        help='Loss function...')

    parser.add_argument(
        '--unit',
        type= str,
        default= "month", #week, week_lognorm
        metavar='UN',
        help='month, week, week_lognorm...')
    
    parser.add_argument(
        '--gpu_id',
        type= int,
        default= 0, #week, week_lognorm
        help='0, 1')


    args = parser.parse_args()

    # Handling string inputs
    if type(args.aggregators) != list:
        args.aggregators = args.aggregators.split()

    if type(args.scalers) != list:
        args.scalers = args.scalers.split()

    # This can be used to print all parser arguments
    parser_args_list = sorted(vars(args).keys())
    args.str = "-".join([f"{arg}:{str(vars(args)[arg])}" for arg in parser_args_list])

    return args


def create_directories(lst_path):
    """Create nested directories for the input paths
    
    Args:
        lst_path (list): file paths

    """
    for path in lst_path:
        Path(path).mkdir(parents=True, exist_ok=True)


def extract_LT(list):
    unique_elements = set(list)
    ToIndex = {}
    ToElement = {}
    for i, element in enumerate(unique_elements):
        ToIndex[str(element)] = i
        ToElement[str(i)] = element

    return (ToIndex,ToElement)


def convert_wLT(theList,LT) -> list:
    converted_list = list(map(lambda x:LT[str(x)], theList))
    return converted_list


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])


def get_khop_node_score(test_graph, node_id, edgeid_to_mask_dict, n_of_hops):
    subset_nodes, subset_edge_index, mapping, edge_mask = utils.k_hop_subgraph(node_id, n_of_hops, test_graph.edge_index)
    explained_edges = []
    
    total_score=0.0
    for ind, val in enumerate(edge_mask):
        if val.item():
            # print(original_edges[ind])
            n1, n2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
            # if abs(edgeid_to_mask_dict[(n1,n2)] - edgeid_to_mask_dict[(n2,n1)]) < 0.5:
            explained_edges.append((n1,n2))
            total_score += edgeid_to_mask_dict[(n1,n2)]
                # print((n1,n2), edgeid_to_mask_dict[(n1,n2)])
    
    if len(explained_edges)>0:
        total_score = total_score/len(explained_edges)
        return total_score
    else:
        return 0.0


def get_all_k_hop_node_scores(test_graph, edgeid_to_mask_dict, n_of_hops):
    original_graph = utils.to_networkx(test_graph)
    node_list= original_graph.nodes
    nodeid_score_dict = dict()

    for node_id in node_list:
        try:
            node_score = get_khop_node_score(test_graph, node_id, edgeid_to_mask_dict, n_of_hops)
            nodeid_score_dict[node_id] = node_score
        except:

            print("ERROR occured when finding node score!")
            nodeid_score_dict[node_id] = 0.0

    return nodeid_score_dict


def convert_graph_to_anndata(graph, node_id_to_importance_dict, dataset_name, imp_quant_thr=0.90):
    adata = None
    positions = np.array(graph.pos)
    features = np.array(graph.x)
    clinical_type = graph.clinical_type
    img_id= graph.img_id
    p_id= graph.p_id
    tumor_grade= graph.tumor_grade
    osmonth= graph.osmonth
    print(graph)
    cell_type = [str(val[0]) for val in graph.ct_class]
    ct_general = [str(val[1]) for val in graph.ct_class]

    
    obs = [str(val) for val in list(range(graph.x.shape[0]))]
    var = get_gene_list(dataset_name=dataset_name)
    print(var)
    node_importance = []
    for item in obs:
        node_importance.append(node_id_to_importance_dict[int(item)])

    node_importance = np.array(node_importance)
    node_imp_thr = np.quantile(node_importance, imp_quant_thr)

    adata = sc.AnnData(features)
    adata.obs_names = obs
    adata.var_names = var
    adata.obs["clinical_type"] = clinical_type
    adata.obs["img_id"] = str(img_id)
    adata.obs["p_id"] = p_id
    
    adata.obs["tumor_grade"] = str(tumor_grade)
    
    adata.obs["osmonth"] = float(osmonth)
    
    adata.obsm["pos"] = positions
    #print("node_importance", node_importance)
    
    adata.obs["importance"] = node_importance
    
    importances_hard = np.array(node_importance > node_imp_thr, dtype="str")
    # print("importances_hard", importances_hard)
    importances_hard = pd.Series(importances_hard, dtype="category")
    # print(importances_hard)
    adata.obs["importance_hard"] = importances_hard.values
    adata.obs["cell_type"] = cell_type
    adata.obs["class"] = ct_general
    # print(adata.obs)
    # sc.tl.rank_genes_groups(adata, groupby="importance_hard", method='wilcoxon', key_added = f"wilcoxon")
    # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key=f"wilcoxon", show=True, groupby="importance_hard", save="important_vs_unimportant")

    # print(adata.obs)


    # adata.write(os.path.join(OUT_DATA_PATH, "adatafiles", f"{graph.img_id}_{graph.p_id}.h5ad"))
    
    return adata


def get_hvgs(adata):
    pass


def get_gene_list(dataset_name="JacksonFischer"):
    lst_genes = []
    if dataset_name=="JacksonFischer":

        get_id_to_gene_dict = get_basel_zurich_staining_panel()
        lst_features = ['Intensity_MeanIntensity_FullStack_c12',
        'Intensity_MeanIntensity_FullStack_c13',
        'Intensity_MeanIntensity_FullStack_c14',
        'Intensity_MeanIntensity_FullStack_c15',
        'Intensity_MeanIntensity_FullStack_c16',
        'Intensity_MeanIntensity_FullStack_c17',
        'Intensity_MeanIntensity_FullStack_c18',
        'Intensity_MeanIntensity_FullStack_c19',
        'Intensity_MeanIntensity_FullStack_c20',
        'Intensity_MeanIntensity_FullStack_c21',
        'Intensity_MeanIntensity_FullStack_c22',
        'Intensity_MeanIntensity_FullStack_c23',
        'Intensity_MeanIntensity_FullStack_c24',
        'Intensity_MeanIntensity_FullStack_c25',
        'Intensity_MeanIntensity_FullStack_c27',
        'Intensity_MeanIntensity_FullStack_c28',
        'Intensity_MeanIntensity_FullStack_c29',
        'Intensity_MeanIntensity_FullStack_c30',
        'Intensity_MeanIntensity_FullStack_c31',
        'Intensity_MeanIntensity_FullStack_c33',
        'Intensity_MeanIntensity_FullStack_c34',
        'Intensity_MeanIntensity_FullStack_c35',
        'Intensity_MeanIntensity_FullStack_c37',
        'Intensity_MeanIntensity_FullStack_c38',
        'Intensity_MeanIntensity_FullStack_c39',
        'Intensity_MeanIntensity_FullStack_c40',
        'Intensity_MeanIntensity_FullStack_c41',
        'Intensity_MeanIntensity_FullStack_c43',
        'Intensity_MeanIntensity_FullStack_c44',
        'Intensity_MeanIntensity_FullStack_c45',
        'Intensity_MeanIntensity_FullStack_c46',
        'Intensity_MeanIntensity_FullStack_c47',
        'Intensity_MeanIntensity_FullStack_c9']

        lst_genes = [ get_id_to_gene_dict[int(g_id.split("Intensity_MeanIntensity_FullStack_c")[1])] for g_id in lst_features ]
        
    elif dataset_name=="METABRIC":
        
        mean_ion_count_columns = [
        'HH3_total', 'CK19', 'CK8_18', 'Twist', 'CD68', 'CK14', 'SMA', 'Vimentin',
        'c_Myc', 'HER2', 'CD3', 'HH3_ph', 'Erk1_2', 'Slug', 'ER', 'PR', 'p53', 'CD44',
        'EpCAM', 'CD45', 'GATA3', 'CD20', 'Beta_catenin', 'CAIX', 'E_cadherin', 'Ki67',
        'EGFR', 'pS6', 'Sox9', 'vWF_CD31', 'pmTOR', 'CK7', 'panCK', 'c_PARP_c_Casp3',
        'DNA1', 'DNA2', 'H3K27me3', 'CK5', 'Fibronectin'
    ]
        lst_genes = mean_ion_count_columns
    
    return lst_genes


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def p_correlations(data, dim1, dim2, division_dim):
    # Calculate pearson correlation

    print("Pearson correlation between age and survival: ", data[dim2].corr(data[dim1]))

    # Calculate pearson correlation for each division
    for div in data[division_dim].unique():
        print("Pearson correlation between age and survival for division: ", div, data[data[division_dim] == div][dim2].corr(data[data[division_dim] == div][dim1]))

def type_processor(c_data):
    def generate_clinical_type(row):
        ER = row['ER Status']
        HER2 = row['HER2 Status']
        if ER == "Positive" and HER2 == "Positive":
            return "HR+HER2+"
        if ER == "Negative" and HER2 == "Positive":
            return "HR-HER2+"
        if ER == "Positive" and HER2 == "Negative":
            return "HR+HER2-"
        if ER == "Negative" and HER2 == "Negative":
            return "TripleNeg"
        # Check if HER2 is nan
        if HER2 != "Positive" and HER2 != "Negative":
            return "HER2-NAN"
    

    c_data['clinical_type'] = c_data.apply(generate_clinical_type, axis=1)
    return c_data



def split_by_group(dataset, random_state=42):
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit 

    lst_groups = []
    for item in dataset:
        lst_groups.append([item.p_id, item.img_id, item.clinical_type, item.tumor_grade, item.osmonth])
    df_dataset = pd.DataFrame(lst_groups, columns=["p_id", "img_id", "clinical_type", "tumor_grade", "osmonth"])
    
    splitter = GroupShuffleSplit(test_size=.10, n_splits=2, random_state = random_state)
    split = splitter.split(df_dataset, groups=df_dataset['p_id'])
    trainval_inds, test_inds = next(split)
    # print(trainval_inds)
    trainval = df_dataset.iloc[trainval_inds]
    test = df_dataset.iloc[test_inds]
    
    splitter2 = GroupShuffleSplit(test_size=.10, n_splits=2, random_state = random_state)
    split2 = splitter2.split(trainval, groups=trainval['p_id'])
    train_inds, val_inds = next(split2)

    train = trainval.iloc[train_inds]
    validation = trainval.iloc[val_inds]

    # print("Split by group dataframe", set(train.index)&set(validation.index), set(train.index)&set(test.index))
    assert len(set(train["p_id"])&set(validation["p_id"]))==0 and len(set(train["p_id"])&set(test["p_id"]))==0 and len(set(test["p_id"])&set(validation["p_id"]))==0


    return list(train.index), list(validation.index), list(test.index)
 

def get_n_fold_split(dataset, dataset_name):
    
    json_fl = load_json(f"../data/{dataset_name}/folds.json")
    # json_fl = load_json("../models/METABRIC_GATV2_CoxPHLoss_10_fold_gpusaez_14-04-2024/-pEmEqMbmY3nybm13c6U6Q.json")
    # json_fl = load_json("/home/rifaioglu/projects/GNNClinicalOutcomePrediction/models/JacksonFischer_Final/jswXq05BE0Bx2c29gNGEMg.json")
    lst_groups = []
    for ind, item in enumerate(dataset):
        lst_groups.append([item.p_id, item.img_id, item.clinical_type, item.tumor_grade, item.osmonth])

    df_dataset = pd.DataFrame(lst_groups, columns=["p_id", "img_id", "clinical_type", "tumor_grade", "osmonth"])
    

    
    # List to save sampler triplet
    samplers = []

    for fold in json_fl["fold_img_id_dict"]:
        train_idx = list(df_dataset.loc[df_dataset.img_id.isin(json_fl["fold_img_id_dict"][fold][0]),:].index)
        test_idx = list(df_dataset.loc[df_dataset.img_id.isin(json_fl["fold_img_id_dict"][fold][1]),:].index)
    
        samplers.append((
                (fold), # fold number
                (torch.utils.data.SubsetRandomSampler(train_idx)),
                (torch.utils.data.SubsetRandomSampler(test_idx))))


    return samplers


def extract_patient_id(sample_id):
    """
    Extract patient ID from sample ID.
    
    Parameters:
    -----------
    sample_id : str
        Sample ID in format like "LUAD_D352_ur" or "LUAD_D352ur_LUAD_D352"
        
    Returns:
    --------
    str : Patient ID (e.g., "LUAD_D352")
    """
    # Handle different sample ID formats
    if "_" in sample_id:
        parts = sample_id.split("_")
        
        # Check if it's the format "LUAD_D352ur_LUAD_D352" (4 parts)
        if len(parts) == 4:
            # Take the last two parts: "LUAD_D352"
            return "_".join(parts[-2:])
        elif len(parts) >= 2:
            # For format like "LUAD_D352_ur" -> "LUAD_D352" (take all except last)
            return "_".join(parts[:-1])
    
    # If no underscore or other format, return as is
    return sample_id

def create_stratified_cv_folds_lung_dataset(dataset, clinical_outcome_label, n_folds=5, random_state=42):
    """
    Create stratified cross-validation folds with equal class representation and patient-level stratification.
    FIXED VERSION - No duplicate indices.
    
    Parameters:
    -----------
    dataset : PyTorch Dataset
        The dataset containing samples with clinical outcomes
    clinical_outcome_label : str
        The clinical outcome of interest (e.g., "Progression")
    n_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary where keys are fold numbers (0 to n_folds-1) and values are 
           lists containing [train_indices, validation_indices]
    """
    
    # Extract sample information
    sample_data = []
    
    for idx, data in enumerate(dataset):
        sample_id = data.sample_id
        # Extract patient ID using the helper function
        patient_id = extract_patient_id(sample_id)
        
        sample_data.append({
            'sample_idx': idx,
            'sample_id': sample_id,
            'patient_id': patient_id,
            'outcome': data.y.item() if hasattr(data.y, 'item') else data.y
        })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(sample_data)
    # print(df)
    # Count class distribution
    class_counts = df['outcome'].value_counts()
    """print(f"Original class distribution:")
    for class_val, count in class_counts.items():
        print(f"  Class {class_val}: {count} samples")"""
    
    # Find the minority class count
    minority_class_count = min(class_counts.values)
    # print(f"Minority class count: {minority_class_count}")
    
    # Get samples for each class
    class_0_samples = df[df['outcome'] == 0].copy()
    class_1_samples = df[df['outcome'] == 1].copy()
    
    # Downsample majority class to match minority class
    if len(class_0_samples) > minority_class_count:
        class_0_samples = class_0_samples.sample(n=minority_class_count, random_state=random_state)
    if len(class_1_samples) > minority_class_count:
        class_1_samples = class_1_samples.sample(n=minority_class_count, random_state=random_state)
    
    # Combine balanced samples
    balanced_df = pd.concat([class_0_samples, class_1_samples]).reset_index(drop=True)
    
    # print(f"Balanced class distribution:")
    balanced_class_counts = balanced_df['outcome'].value_counts()
    """for class_val, count in balanced_class_counts.items():
        print(f"  Class {class_val}: {count} samples")"""
    
    # Create patient-level stratification
    # Get unique patients and their outcomes
    patient_outcomes = balanced_df.groupby('patient_id')['outcome'].first().reset_index()
    
    # Create stratified folds at patient level
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds_dict = {}
    samplers = []
    for fold_idx, (train_patient_indices, val_patient_indices) in enumerate(skf.split(patient_outcomes['patient_id'], patient_outcomes['outcome'])):
        
        # Get patient IDs for this fold
        train_patients = patient_outcomes.iloc[train_patient_indices]['patient_id'].tolist()
        val_patients = patient_outcomes.iloc[val_patient_indices]['patient_id'].tolist()
        
        # Get sample indices for training and validation
        train_sample_indices = balanced_df[balanced_df['patient_id'].isin(train_patients)]['sample_idx'].tolist()
        val_sample_indices = balanced_df[balanced_df['patient_id'].isin(val_patients)]['sample_idx'].tolist()
        samplers.append((
                (fold_idx), # fold number
                (torch.utils.data.SubsetRandomSampler(train_sample_indices)),
                (torch.utils.data.SubsetRandomSampler(val_sample_indices))))

        folds_dict[fold_idx] = [train_sample_indices, val_sample_indices]
        
        # Print fold statistics
        train_outcomes = balanced_df[balanced_df['sample_idx'].isin(train_sample_indices)]['outcome']
        val_outcomes = balanced_df[balanced_df['sample_idx'].isin(val_sample_indices)]['outcome']
        
        """print(f"\nFold {fold_idx}:")
        print(f"  Training: {len(train_sample_indices)} samples from {len(train_patients)} patients")
        print(f"  Validation: {len(val_sample_indices)} samples from {len(val_patients)} patients")
        print(f"  Training class distribution: {train_outcomes.value_counts().to_dict()}")
        print(f"  Validation class distribution: {val_outcomes.value_counts().to_dict()}")"""
    
    


    return samplers

def get_train_val_test_split_by_pid(dataset, dataset_name):
    import pandas as pd
    from dataset import TissueDataset
    # dataset = TissueDataset(os.path.join("../data/JacksonFischer", "month"),  "month")

    lst_groups = []
    for item in dataset:
        lst_groups.append([item.p_id, item.img_id, item.clinical_type, item.tumor_grade, item.osmonth])

    
    df_dataset = pd.DataFrame(lst_groups, columns=["p_id", "img_id", "clinical_type", "tumor_grade", "osmonth"])
    df_dataset_with_splits = None
    if dataset_name=="METABRIC":
        df_dataset_with_splits = pd.read_csv("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/plots/regression_split/METABRIC_47_rest.csv")
    elif dataset_name=="JacksonFischer":
        df_dataset_with_splits = pd.read_csv("/net/data.isilon/ag-saez/bq_arifaioglu/home/Projects/GNNClinicalOutcomePrediction/plots/regression_split/JacksonFischer_89_rest.csv")
    else:
        raise ValueError("Invalid dataset name!")

    train_img_ids = df_dataset_with_splits[df_dataset_with_splits["tvt"].str.contains("train")]["img_id"].values
    val_img_ids = df_dataset_with_splits[df_dataset_with_splits["tvt"].str.contains("val")]["img_id"].values
    test_img_ids = df_dataset_with_splits[df_dataset_with_splits["tvt"].str.contains("test")]["img_id"].values
    train = df_dataset[df_dataset["img_id"].isin(train_img_ids)]
    validation = df_dataset[df_dataset["img_id"].isin(val_img_ids)]
    test = df_dataset[df_dataset["img_id"].isin(test_img_ids)]
    # print(sorted(list(df_dataset[df_dataset["img_id"].isin(train_img_ids)]["img_id"])))
    # print(sorted(list(df_dataset_with_splits[df_dataset_with_splits["tvt"].str.contains("train")]["img_id"])))
    assert sorted(list(df_dataset[df_dataset["img_id"].isin(train_img_ids)]["img_id"]))==sorted(list(df_dataset_with_splits[df_dataset_with_splits["tvt"].str.contains("train")]["img_id"]))
    assert len(set(train["p_id"])&set(validation["p_id"]))==0 and len(set(train["p_id"])&set(test["p_id"]))==0 and len(set(test["p_id"])&set(validation["p_id"]))==0
    return list(train.index), list(validation.index), list(test.index)

# get_train_val_test_split_by_pid()


def clean_session_files(result_fold_path, model_fold_path, model_id, gnn_layer):
    pathlib.Path.unlink( Path(os.path.join(model_fold_path, f"{model_id}_SD.mdl")))
    pathlib.Path.unlink(Path(os.path.join(model_fold_path, f"{model_id}.json")))
    if gnn_layer=="PNAConv":
        pathlib.Path.unlink(Path(os.path.join(model_fold_path, f"{model_id}_deg.pckl")))
    pathlib.Path.unlink(Path(os.path.join(result_fold_path, f"{model_id}.csv")))
    
    
def generate_splits(dataset_name):
    from dataset import TissueDataset
    from matplotlib import pyplot as plt
    dataset = TissueDataset(os.path.join(f"../data/{dataset_name}", "month"),  "month")

    lst_groups = []
    for item in dataset:
        print(item)
        lst_groups.append([item.p_id, item.img_id, item.clinical_type, -1 if item.tumor_grade[0].isnan() else int(item.tumor_grade[0].item()), int(item.osmonth[0].item()), int(item.is_censored[0].item())])
    df_dataset = pd.DataFrame(lst_groups, columns=["p_id", "img_id", "clinical_type", "tumor_grade", "osmonth", "is_censored"])

    for i in range(200):
        train, val, test = split_by_group(dataset, random_state=i)
        
        df_train = df_dataset.iloc[train].copy()
        df_train["tvt"] = "train"
        df_val = df_dataset.iloc[val].copy()
        df_val["tvt"] = "val"
        df_test = df_dataset.iloc[test].copy()
        df_test["tvt"] = "test"
        df_ann_dataset = [df_train, df_val, df_test]
        df_ann_dataset = pd.concat(df_ann_dataset)

        df_train = df_train.drop_duplicates(subset=['p_id'])
        df_val = df_val.drop_duplicates(subset=['p_id'])
        df_test = df_test.drop_duplicates(subset=['p_id'])
        frames = [df_train, df_val, df_test]
        result = pd.concat(frames)

    
        fig, axs = plt.subplots(1,3,figsize=(20,5))
        
        # print(result.groupby(['tvt', "clinical_type"]).count())
        df_train.groupby(["clinical_type"]).count().plot( kind='pie', y = "osmonth", autopct='%1.0f%%', ax = axs[0])
        df_val.groupby(["clinical_type"]).count().plot( kind='pie', y = "osmonth", autopct='%1.0f%%', ax = axs[1])
        df_test.groupby(["clinical_type"]).count().plot( kind='pie', y = "osmonth", autopct='%1.0f%%', ax = axs[2]) 
        plt.savefig(f"../plots/regression_split/{dataset_name}_{i}_pie_rest.png")

        axes = result.boxplot("osmonth", by="tvt")
        fig = axes.get_figure()
        print(sum(df_train["is_censored"])/len(df_train))

        df_train = df_dataset.iloc[train].copy()
        df_train["tvt"] = "train"
        df_val = df_dataset.iloc[val].copy()
        df_val["tvt"] = "val"
        df_test = df_dataset.iloc[test].copy()
        df_test["tvt"] = "test"
        fig.suptitle(f"Test: {round(1.0-sum(df_test.is_censored)/len(df_test),2)} / Train: {round(1.0-sum(df_train.is_censored)/len(df_train),2)} / Val: {round(1.0-sum(df_val.is_censored)/len(df_val),2)}")
        plt.savefig(f"../plots/regression_split/{dataset_name}_{i}_rest.png")
        plt.close()
        df_ann_dataset.to_csv(f"../plots/regression_split/{dataset_name}_{i}_rest.csv")


def set_seeds(seed=42, deterministic = False):
    pl.seed_everything(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    # 
    np.random.seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        

import numpy as np
from itertools import product

def create_hyperparameter_combinations(hyperparams):
    """
    Create all combinations of hyperparameters from a given dictionary.
    
    Parameters:
    hyperparams (dict): A dictionary where keys are hyperparameter names and values are lists of possible values.
    
    Returns:
    list of dict: A list of dictionaries, each representing a combination of hyperparameters.
    """
    keys = hyperparams.keys()
    values = hyperparams.values()
    
    # Create product of all values
    combinations = list(product(*values))
    
    # Create list of dictionaries for each combination
    combinations_dicts = [dict(zip(keys, combo)) for combo in combinations]
    
    return combinations_dicts