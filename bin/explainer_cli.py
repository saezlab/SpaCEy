import os
import torch
import numpy as np
import custom_tools
import pickle
from torch_geometric import utils
import plotting as plotting
from tqdm import tqdm   
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer
from data_processing import OUT_DATA_PATH
# import anndata as ad
import pytorch_lightning as pl


device = custom_tools.get_device()

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
# RAW_DATA_PATH = os.path.join(S_PATH, "../data", "JacksonFischer/raw")
# RAW_DATA_PATH = os.path.join(S_PATH, "../data", "METABRIC/raw")

    
class Custom_Explainer:
 
    # init method or constructor
    def __init__(self, model, dataset_name, dataset, exp_name, job_id, seed=42):

        custom_tools.set_seeds(seed, deterministic=True)
        self.seed = seed
        self.model = model
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.exp_name = exp_name
        self.job_id = job_id
        self.RAW_DATA_PATH = os.path.join(S_PATH, "../data", f"{dataset_name}/raw")


    def set_dataset(self, dataset):
        self.dataset = dataset

    
    def explain(self,lr: float=0.1,  epoch: int=100, return_type: str = "regression", feat_mask_type: str = "feature"):
        
        
        gene_list = custom_tools.get_gene_list(self.dataset_name)
        print(gene_list)
        adata_concat = []
        count = 0
        for test_graph in tqdm(self.dataset):

            explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode= return_type,
                task_level='graph',
                return_type='raw', 
            ),
            )
            print(test_graph)
            # For jackson fischer dataset
            #with open(os.path.join(self.RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
            with open(os.path.join(self.RAW_DATA_PATH, f'{test_graph.img_id}_coordinates.pickle'), 'rb') as handle:
                coordinates_arr = pickle.load(handle)
        
            # Generate the explanation for a particular graph:
            explanation = explainer(test_graph.x, test_graph.edge_index)
            # print("Edge mask:", explanation.edge_mask)
            # print("Node mask:", explanation.node_mask)
            edge_value_mask = explanation.edge_mask
            quant_thr = 0.80
            
            genename_to_nodeid_dict = dict()

            for col_ind, gene_name in enumerate(gene_list):
                # print(col_ind, gene_name)
                genename_to_nodeid_dict[gene_name] = dict()
                for node_id, val in enumerate(test_graph.x[:,col_ind]):
                    genename_to_nodeid_dict[gene_name][node_id] = val.item()
            
            # print(edge_value_mask.shape)
            # edge_exp_score_mask_arr = np.array(edge_value_mask.cpu())
            edge_exp_score_mask_arr = edge_value_mask.numpy()
            


            # edge_thr = np.quantile(np.array(edge_value_mask.cpu()), quant_thr)
            edge_thr = np.quantile(edge_value_mask.numpy(), quant_thr)

            print(f"Edge thr: {edge_thr:.3f}\tMin: {np.min(edge_exp_score_mask_arr)}\tMax: {np.max(edge_exp_score_mask_arr):.3f}\tMin: {np.min(edge_exp_score_mask_arr):.3f}")
            

            exp_edges_bool = edge_exp_score_mask_arr > edge_thr

            explained_edge_indices = exp_edges_bool.nonzero()[0]

            """for ind, val in enumerate(exp_edges_bool):
                if val:
                    print(val, edge_exp_score_mask_arr[ind])"""
            # print(edge_exp_score_mask_arr)
            # print(list(set(range(len(edge_value_mask)))- set(explained_edge_indices)))
            np.put(edge_exp_score_mask_arr, list(set(range(len(edge_value_mask)))- set(explained_edge_indices)), 0.0)


            edgeid_to_mask_dict = dict()
            for ind, m_val in enumerate(edge_exp_score_mask_arr):
                # print(ind, m_val, exp_edges_bool[ind], edge_value_mask[ind], edge_thr)
                node_id1, node_id2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
                edgeid_to_mask_dict[(node_id1, node_id2)] = m_val.item()
            
            
            n_of_hops = 2   
            # TODO: Check if the scores are calculated over the ccs
            plt.rcParams['figure.figsize'] = 10, 10
            node_to_score_dict = custom_tools.get_all_k_hop_node_scores(test_graph, edgeid_to_mask_dict, n_of_hops)

            adata = custom_tools.convert_graph_to_anndata(test_graph, node_to_score_dict, dataset_name=self.dataset_name)
            #adata_concat.append(adata)
            plt.rcParams['figure.figsize'] = 48, 8
            fig, axs = plt.subplots(1, 4)
            plotting.plot_graph(test_graph, coordinates_arr, axs[0], font_size=5,  node_size=100, width=1)
            plotting.plot_node_importances(test_graph, coordinates_arr, node_to_score_dict,  axs[2], node_size=100, width=1)
            plotting.plot_node_importances_voronoi(test_graph, coordinates_arr, node_to_score_dict,  axs[1])
            # plotting.plot_node_types_voronoi(test_graph, coordinates_arr, axs[3])
            # plotting.plot_node_importances_voronoi(test_graph, coordinates_arr,  genename_to_nodeid_dict[gene_list[0]],  axs[0][3], title=gene_list[0], cmap=plt.cm.GnBu)

            """cols = 4
            for ind,val in enumerate(gene_list[1:]):
                fig_row, fig_col = int(ind/cols), ind%cols
                plotting.plot_node_importances_voronoi(test_graph, coordinates_arr,  genename_to_nodeid_dict[gene_list[ind+1]],  axs[fig_row+1][fig_col], title=gene_list[ind+1], cmap=plt.cm.GnBu)"""

            plot = True
            if plot:
                if self.dataset_name!="Lung":
                    fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.p_id}_{str(int(test_graph.osmonth))}_{test_graph.clinical_type}.png"))
                    fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.p_id}_{str(int(test_graph.osmonth))}_{test_graph.clinical_type}.pdf"))
                else:
                    fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.y}.png"))
                    fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.y}.pdf"))
                plt.close()
            count +=1
            adata.write(os.path.join(OUT_DATA_PATH, "adatafiles", self.dataset_name, f"{self.exp_name}_{self.job_id}_{test_graph.img_id}.h5ad"))
        
            #if count ==3:
            #   break

        #print(adata_concat)
        
        """adata = ad.concat(adata_concat)
        print(adata)
        adata.write(os.path.join(OUT_DATA_PATH, "adatafiles", self.dataset_name, f"{self.exp_name}_{self.job_id}_concatenated_explanations.h5ad"))"""



    def explain_by_gnnexplainer(self,lr: float=0.1,  epoch: int=100, return_type: str = "regression", feat_mask_type: str = "feature"):
        """
        Learns and returns a node feature mask and an edge mask that play a crucial role to explain the prediction made by the GNN for a graph.
        test_graph.x (torch.Tensor): The node feature matrix
        test_graph.edge_index (torch.Tensor): The edge indices.
        
        Additional hyper-parameters to override default settings in coeffs.
        coeffs = {'edge_ent': 1.0, 'edge_reduction': 'sum', 'edge_size': 0.005, 'node_feat_ent': 0.1, 'node_feat_reduction': 'mean', 'node_feat_size': 1.0}
        """

        gene_list = custom_tools.get_gene_list()
        adata_concat = []
        count = 0
        for test_graph in tqdm(self.dataset):
            explainer = GNNExplainer(self.model, epochs = epoch, lr = lr,
                                    return_type = return_type, feat_mask_type = feat_mask_type).to(device)    
            print(f"Sample id: {test_graph.img_id}_{test_graph.p_id}")
            with open(os.path.join(RAW_DATA_PATH, f'{test_graph.img_id}_{test_graph.p_id}_coordinates.pickle'), 'rb') as handle:
                coordinates_arr = pickle.load(handle)
            
            # number of nodes
            # test_graph.num_nodes g.number_of_nodes() g.number_of_edges()

            (feature_val_mask, edge_value_mask) = explainer.explain_graph(test_graph.x.to(device), test_graph.edge_index.to(device))
            quant_thr = 0.80
            
            genename_to_nodeid_dict = dict()

            for col_ind, gene_name in enumerate(gene_list):
                genename_to_nodeid_dict[gene_name] = dict()
                for node_id, val in enumerate(test_graph.x[:,col_ind]):
                    genename_to_nodeid_dict[gene_name][node_id] = val.item()
            
        
            edge_exp_score_mask_arr = np.array(edge_value_mask.cpu())


            edge_thr = np.quantile(np.array(edge_value_mask.cpu()), quant_thr)

            # print(f"Edge thr: {edge_thr:.3f}\tMin: {np.min(edge_exp_score_mask_arr)}\tMax: {np.max(edge_exp_score_mask_arr):.3f}\tMin: {np.min(edge_exp_score_mask_arr):.3f}")
            

            exp_edges_bool = edge_exp_score_mask_arr > edge_thr

            explained_edge_indices = exp_edges_bool.nonzero()[0]
            np.put(edge_exp_score_mask_arr, list(set(range(len(edge_value_mask)))- set(explained_edge_indices)), 0.0)


            edgeid_to_mask_dict = dict()
            for ind, m_val in enumerate(edge_exp_score_mask_arr):
                node_id1, node_id2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
                edgeid_to_mask_dict[(node_id1, node_id2)] = m_val.item()
            
            
            n_of_hops = 2   
            # TODO: Check if the scores are calculated over the ccs
            plt.rcParams['figure.figsize'] = 10, 10
            node_to_score_dict = custom_tools.get_all_k_hop_node_scores(test_graph, edgeid_to_mask_dict, n_of_hops)

            adata = custom_tools.convert_graph_to_anndata(test_graph, node_to_score_dict)
            
            plt.rcParams['figure.figsize'] = 50, 100
            fig, axs = plt.subplots(9, 4)
            plotting.plot_graph(test_graph, coordinates_arr, axs[0][0], font_size=5,  node_size=100, width=1)
            plotting.plot_node_importances(test_graph, coordinates_arr, node_to_score_dict,  axs[0][2], node_size=100, width=1)
            plotting.plot_node_importances_voronoi(test_graph, coordinates_arr, node_to_score_dict,  axs[0][1])
            plotting.plot_node_importances_voronoi(test_graph, coordinates_arr,  genename_to_nodeid_dict[gene_list[0]],  axs[0][3], title=gene_list[0], cmap=plt.cm.GnBu)
            
            cols = 4
            for ind,val in enumerate(gene_list[1:]):
                fig_row, fig_col = int(ind/cols), ind%cols
                plotting.plot_node_importances_voronoi(test_graph, coordinates_arr,  genename_to_nodeid_dict[gene_list[ind+1]],  axs[fig_row+1][fig_col], title=gene_list[ind+1], cmap=plt.cm.GnBu)


            fig.savefig(f"../plots/subgraphs/{test_graph.img_id}_{test_graph.p_id}_{str(int(test_graph.osmonth))}_{test_graph.clinical_type}")
            plt.close()
        
            count +=1
            # if count ==10:
            #     break
        # adata = adata_concat[0].concatenate(adata_concat[1:], join='outer')
        # adata.write(os.path.join(OUT_DATA_PATH, "adatafiles", f"concatenated_explanations.h5ad"))

        



 
 


# python gnnexplainer.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 1