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
    """
    Custom explainer class for GNN models with evaluation capabilities.
    
    Example usage with per-sample learning rate selection:
        explainer = Custom_Explainer(model, dataset_name, dataset, exp_name, job_id, seed=42)
        
        # Per-sample LR selection: Each sample gets its own best LR
        # For each sample:
        #   1. For each LR, try all seeds and calculate average score
        #   2. Select LR with best average score
        #   3. Use best seed for selected LR
        results = explainer.explain_with_evaluation(
            lr_list=[0.01, 0.1, 0.5],
            seed_list=[42, 123, 456],  # Multiple seeds - averaged for each LR
            evaluation_metric='fidelity'
        )
    """
 
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

    
    def evaluate_explanation(self, explanation, test_graph, return_type: str = "regression"):
        """
        Evaluate the quality of an explanation using multiple metrics.
        
        Returns:
            dict: Dictionary containing evaluation metrics:
                - fidelity: How well the explanation matches the model's prediction
                - sparsity: Fraction of edges/nodes kept in explanation
                - edge_mask_entropy: Entropy of edge mask (higher = more uniform)
        """
        self.model.eval()
        
        # Get original prediction
        # For graph-level tasks, need batch parameter
        batch = torch.zeros(test_graph.x.shape[0], dtype=torch.long, device=test_graph.x.device)
        with torch.no_grad():
            original_out = self.model(test_graph.x, test_graph.edge_index, batch=batch)
            if return_type == "regression":
                original_pred = original_out.item() if original_out.numel() == 1 else original_out
            else:
                # For classification, get the predicted class
                if original_out.dim() == 0:
                    original_pred = original_out.item()
                elif original_out.numel() == 1:
                    original_pred = original_out.item()
                else:
                    original_pred = original_out.argmax(dim=-1)
                    # If result is still a tensor, convert to scalar
                    if isinstance(original_pred, torch.Tensor):
                        original_pred = original_pred.item() if original_pred.numel() == 1 else original_pred[0].item()
        
        # Get prediction with explanation mask applied
        edge_mask = explanation.edge_mask
        node_mask = explanation.node_mask if hasattr(explanation, 'node_mask') else None
        
        # Apply masks
        if node_mask is not None:
            masked_x = test_graph.x * node_mask.sigmoid()
        else:
            masked_x = test_graph.x
        
        # Create masked edge_index (keep only important edges)
        # Use median threshold to keep top 50% of edges
        if isinstance(edge_mask, torch.Tensor):
            edge_threshold = edge_mask.quantile(0.5).item()
            important_edges = edge_mask > edge_threshold
        else:
            # If numpy array
            edge_threshold = np.quantile(edge_mask, 0.5)
            important_edges = edge_mask > edge_threshold
        
        # Ensure important_edges is boolean and has correct shape
        if isinstance(important_edges, torch.Tensor):
            important_edges = important_edges.bool()
        else:
            important_edges = torch.tensor(important_edges, dtype=torch.bool)
        
        # Filter edge_index to keep only important edges
        if important_edges.sum() > 0:
            masked_edge_index = test_graph.edge_index[:, important_edges]
        else:
            # If no edges selected, use empty edge_index
            masked_edge_index = torch.empty((2, 0), dtype=test_graph.edge_index.dtype, device=test_graph.edge_index.device)
        
        # Create batch for masked graph (all nodes belong to same graph)
        masked_batch = torch.zeros(masked_x.shape[0], dtype=torch.long, device=masked_x.device)
        with torch.no_grad():
            masked_out = self.model(masked_x, masked_edge_index, batch=masked_batch)
            if return_type == "regression":
                if masked_out.numel() == 1:
                    masked_pred = masked_out.item()
                else:
                    masked_pred = masked_out
            else:
                # For classification, get the predicted class
                if masked_out.dim() == 0:
                    masked_pred = masked_out.item()
                elif masked_out.numel() == 1:
                    masked_pred = masked_out.item()
                else:
                    masked_pred = masked_out.argmax(dim=-1)
                    # If result is still a tensor, convert to scalar
                    if isinstance(masked_pred, torch.Tensor):
                        masked_pred = masked_pred.item() if masked_pred.numel() == 1 else masked_pred[0].item()
        
        # Calculate fidelity (how similar are predictions)
        if return_type == "regression":
            if isinstance(original_pred, torch.Tensor):
                original_pred_val = original_pred.item() if original_pred.numel() == 1 else original_pred
            else:
                original_pred_val = original_pred
            
            if isinstance(masked_pred, torch.Tensor):
                masked_pred_val = masked_pred.item() if masked_pred.numel() == 1 else masked_pred
            else:
                masked_pred_val = masked_pred
            
            abs_diff = abs(original_pred_val - masked_pred_val)
            abs_original = abs(original_pred_val) + 1e-8
            fidelity = 1.0 - (abs_diff / abs_original)
            fidelity = max(0.0, min(1.0, fidelity))  # Clamp between 0 and 1
        else:
            # For classification, compare predictions (should be scalars at this point)
            # Convert to Python int/float for comparison
            if isinstance(original_pred, torch.Tensor):
                original_pred_val = original_pred.item()
            else:
                original_pred_val = int(original_pred) if isinstance(original_pred, (int, float, np.integer)) else original_pred
            
            if isinstance(masked_pred, torch.Tensor):
                masked_pred_val = masked_pred.item()
            else:
                masked_pred_val = int(masked_pred) if isinstance(masked_pred, (int, float, np.integer)) else masked_pred
            
            # Compare scalar predictions
            fidelity = float(original_pred_val == masked_pred_val)
        
        # Calculate sparsity (fraction of edges kept)
        if isinstance(important_edges, torch.Tensor):
            sparsity = important_edges.float().mean().item()
        else:
            sparsity = important_edges.mean()
        
        # Calculate entropy of edge mask (measure of concentration)
        if isinstance(edge_mask, torch.Tensor):
            edge_mask_probs = edge_mask.sigmoid()
            eps = 1e-8
            entropy = -(edge_mask_probs * torch.log(edge_mask_probs + eps) + 
                       (1 - edge_mask_probs) * torch.log(1 - edge_mask_probs + eps)).mean().item()
            edge_mask_mean = edge_mask.mean().item()
            edge_mask_std = edge_mask.std().item()
        else:
            # Convert numpy to tensor for sigmoid
            edge_mask_tensor = torch.tensor(edge_mask, dtype=torch.float32)
            edge_mask_probs = edge_mask_tensor.sigmoid()
            eps = 1e-8
            entropy = -(edge_mask_probs * torch.log(edge_mask_probs + eps) + 
                       (1 - edge_mask_probs) * torch.log(1 - edge_mask_probs + eps)).mean().item()
            edge_mask_mean = float(np.mean(edge_mask))
            edge_mask_std = float(np.std(edge_mask))
        
        return {
            'fidelity': fidelity,
            'sparsity': sparsity,
            'entropy': entropy,
            'edge_mask_mean': edge_mask_mean,
            'edge_mask_std': edge_mask_std,
        }
    
    def explain_single(self, test_graph, lr: float=0.1, epoch: int=200, 
                      return_type: str = "regression", seed: int = None):
        """
        Generate explanation for a single graph with specific hyperparameters.
        
        Returns:
            explanation: The explanation object
            eval_metrics: Dictionary of evaluation metrics
        """
        if seed is not None:
            custom_tools.set_seeds(seed, deterministic=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=epoch, lr=lr),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode=return_type,
                task_level='graph',
                return_type='raw', 
            ),
        )
        
        # Generate the explanation
        # For graph-level explanations, need to pass batch parameter
        batch = torch.zeros(test_graph.x.shape[0], dtype=torch.long, device=test_graph.x.device)
        explanation = explainer(test_graph.x, test_graph.edge_index, batch=batch)
        
        # Debug: Check edge mask values
        edge_mask = explanation.edge_mask
        if isinstance(edge_mask, torch.Tensor):
            print(f"Edge mask stats - Min: {edge_mask.min().item():.6f}, Max: {edge_mask.max().item():.6f}, Mean: {edge_mask.mean().item():.6f}, Non-zero: {(edge_mask != 0).sum().item()}/{edge_mask.numel()}")
        else:
            edge_mask_np = np.array(edge_mask)
            print(f"Edge mask stats - Min: {edge_mask_np.min():.6f}, Max: {edge_mask_np.max():.6f}, Mean: {edge_mask_np.mean():.6f}, Non-zero: {np.count_nonzero(edge_mask_np)}/{edge_mask_np.size}")
        
        # Evaluate the explanation
        eval_metrics = self.evaluate_explanation(explanation, test_graph, return_type)
        
        return explanation, eval_metrics
    
    def explain_with_evaluation(self, lr: float=0.1, epoch: int=200, 
                                return_type: str = "regression", 
                                lr_list: list = None, seed_list: list = None,
                                evaluation_metric: str = 'fidelity'):
        """
        Run explanations with multiple configurations and select the best one per sample.
        
        For each sample:
        1. For each learning rate, try all seeds and calculate average score across seeds
        2. Select the learning rate with the best average score
        3. Use the best seed for the selected learning rate
        4. Save only the best LR-seed combination
        
        Args:
            lr: Default learning rate (used if lr_list is None)
            epoch: Number of epochs
            return_type: Type of prediction task
            lr_list: List of learning rates to try (e.g., [0.01, 0.1, 0.5])
            seed_list: List of random seeds to try (e.g., [42, 123, 456])
            evaluation_metric: Metric to use for selection ('fidelity', 'sparsity', or 'combined')
        
        Returns:
            dict: Best configuration and its metrics for each sample
        """
        if lr_list is None:
            lr_list = [lr]
        if seed_list is None:
            seed_list = [self.seed]
        
        gene_list = custom_tools.get_gene_list(self.dataset_name)
        print(f"Evaluating explanations with {len(lr_list)} learning rate(s) and {len(seed_list)} seed(s)")
        print(f"Each sample will get its own best LR selected based on average scores across seeds")
        
        results_summary = []
        
        for test_graph in tqdm(self.dataset, desc="Processing samples"):
            # For each LR, collect scores across all seeds
            lr_scores = {lr_val: [] for lr_val in lr_list}
            lr_configs = {lr_val: [] for lr_val in lr_list}  # Store all configs for each LR
            
            # Try all seeds for each learning rate
            for lr_val in lr_list:
                for seed_val in seed_list:
                    try:
                        explanation, eval_metrics = self.explain_single(
                            test_graph, lr=lr_val, epoch=epoch, 
                            return_type=return_type, seed=seed_val
                        )
                        
                        # Calculate composite score
                        if evaluation_metric == 'fidelity':
                            score = eval_metrics['fidelity']
                        elif evaluation_metric == 'sparsity':
                            score = eval_metrics['sparsity']
                        elif evaluation_metric == 'combined':
                            score = eval_metrics['fidelity'] * 0.7 + eval_metrics['sparsity'] * 0.3
                        else:
                            score = eval_metrics.get(evaluation_metric, eval_metrics['fidelity'])
                        
                        config_info = {
                            'lr': lr_val,
                            'seed': seed_val,
                            'score': score,
                            'metrics': eval_metrics,
                            'explanation': explanation
                        }
                        
                        # Store score and config for this LR
                        lr_scores[lr_val].append(score)
                        lr_configs[lr_val].append(config_info)
                    
                    except Exception as e:
                        print(f"Error with lr={lr_val}, seed={seed_val} for {test_graph.img_id}: {e}")
                        continue
            
            # Calculate average score for each LR (across all seeds)
            lr_avg_scores = {}
            for lr_val in lr_list:
                if len(lr_scores[lr_val]) > 0:
                    lr_avg_scores[lr_val] = np.mean(lr_scores[lr_val])
                else:
                    lr_avg_scores[lr_val] = -float('inf')
            
            # Select LR with best average score
            best_lr = max(lr_avg_scores.keys(), key=lambda lr: lr_avg_scores[lr])
            
            # For the best LR, select the seed with the best individual score
            best_config = None
            best_score = -float('inf')
            best_explanation = None
            
            for config in lr_configs[best_lr]:
                if config['score'] > best_score:
                    best_score = config['score']
                    best_config = config
                    best_explanation = config['explanation']
            
            # Print per-sample LR selection summary
            if len(lr_list) > 1:
                print(f"\n{'='*80}")
                print(f"Sample: {test_graph.img_id}")
                print(f"{'='*80}")
                print(f"{'LR':<10} {'Avg Score':<15} {'Seeds Tested':<15} {'Best Seed Score':<15} {'Status':<10}")
                print("-"*80)
                for lr_val in sorted(lr_list, key=lambda lr: lr_avg_scores[lr], reverse=True):
                    marker = " <-- SELECTED" if lr_val == best_lr else ""
                    n_seeds = len(lr_scores[lr_val])
                    best_seed_score = max(lr_scores[lr_val]) if n_seeds > 0 else 0.0
                    avg_score = lr_avg_scores[lr_val]
                    print(f"{lr_val:<10.4f} {avg_score:<15.4f} {n_seeds:<15} {best_seed_score:<15.4f}{marker}")
                print(f"{'='*80}")
                print(f"Selected LR: {best_lr} (avg score: {lr_avg_scores[best_lr]:.4f})")
                print(f"Best seed for selected LR: {best_config['seed']} (score: {best_score:.4f})")
                print(f"{'='*80}")
            
            if best_config is None:
                print(f"Warning: No valid explanation found for graph {test_graph.img_id}")
                continue
            
            # Process best explanation (same as original explain method)
            with open(os.path.join(self.RAW_DATA_PATH, f'{test_graph.img_id}_coordinates.pickle'), 'rb') as handle:
                coordinates_arr = pickle.load(handle)
            
            edge_value_mask = best_explanation.edge_mask
            quant_thr = 0.80
            # Ensure edge mask is on CPU and detached before converting to numpy
            if isinstance(edge_value_mask, torch.Tensor):
                edge_exp_score_mask_arr = edge_value_mask.detach().cpu().numpy()
            else:
                edge_exp_score_mask_arr = np.array(edge_value_mask)
            edge_thr = np.quantile(edge_exp_score_mask_arr, quant_thr)
            
            # Debug: Print edge mask statistics before thresholding
            print(f"Edge mask before thresholding - Min: {edge_exp_score_mask_arr.min():.6f}, Max: {edge_exp_score_mask_arr.max():.6f}, Mean: {edge_exp_score_mask_arr.mean():.6f}, Threshold (80th percentile): {edge_thr:.6f}")
            
            exp_edges_bool = edge_exp_score_mask_arr > edge_thr
            explained_edge_indices = exp_edges_bool.nonzero()[0]
            
            # Debug: Print how many edges are above threshold
            print(f"Edges above threshold: {len(explained_edge_indices)}/{len(edge_exp_score_mask_arr)} ({100*len(explained_edge_indices)/len(edge_exp_score_mask_arr):.1f}%)")
            
            # Set edges below threshold to 0, but keep original values for edges above threshold
            edge_exp_score_mask_arr_thresholded = edge_exp_score_mask_arr.copy()
            edge_exp_score_mask_arr_thresholded[~exp_edges_bool] = 0.0
            
            # Use the thresholded array for further processing
            edge_exp_score_mask_arr = edge_exp_score_mask_arr_thresholded
            
            # Debug: Print edge mask statistics after thresholding
            print(f"Edge mask after thresholding - Non-zero: {np.count_nonzero(edge_exp_score_mask_arr)}/{len(edge_exp_score_mask_arr)}, Mean of non-zero: {edge_exp_score_mask_arr[edge_exp_score_mask_arr > 0].mean():.6f}")
            
            edgeid_to_mask_dict = dict()
            for ind, m_val in enumerate(edge_exp_score_mask_arr):
                node_id1, node_id2 = test_graph.edge_index[0,ind].item(), test_graph.edge_index[1,ind].item()
                edgeid_to_mask_dict[(node_id1, node_id2)] = float(m_val)  # Convert to Python float
            
            n_of_hops = 2
            plt.rcParams['figure.figsize'] = 10, 10
            node_to_score_dict = custom_tools.get_all_k_hop_node_scores(test_graph, edgeid_to_mask_dict, n_of_hops)
            
            adata = custom_tools.convert_graph_to_anndata(test_graph, node_to_score_dict, dataset_name=self.dataset_name)
            
            # Save plot with best config info in filename
            best_lr = best_config['lr']
            best_seed = best_config['seed']
            plt.rcParams['figure.figsize'] = 48, 8
            fig, axs = plt.subplots(1, 4)
            plotting.plot_graph(test_graph, coordinates_arr, axs[0], font_size=5, node_size=100, width=1)
            plotting.plot_node_importances(test_graph, coordinates_arr, node_to_score_dict, axs[2], node_size=100, width=1)
            plotting.plot_node_importances_voronoi(test_graph, coordinates_arr, node_to_score_dict, axs[1])
            
            plot = True
            if plot:
                filename_suffix = f"lr{best_lr}_seed{best_seed}_fid{best_config['metrics']['fidelity']:.3f}"
                if self.dataset_name != "Lung":
                    fig.savefig(os.path.join("../plots/explanations", self.dataset_name, 
                                           f"{self.exp_name}_{self.job_id}", 
                                           f"{test_graph.img_id}_{test_graph.p_id}_{str(int(test_graph.osmonth))}_{test_graph.clinical_type}_{filename_suffix}.png"))
                    fig.savefig(os.path.join("../plots/explanations", self.dataset_name, 
                                           f"{self.exp_name}_{self.job_id}", 
                                           f"{test_graph.img_id}_{test_graph.p_id}_{str(int(test_graph.osmonth))}_{test_graph.clinical_type}_{filename_suffix}.pdf"))
                else:
                    fig.savefig(os.path.join("../plots/explanations", self.dataset_name, 
                                           f"{self.exp_name}_{self.job_id}", 
                                           f"{test_graph.img_id}_{test_graph.y}_{filename_suffix}.png"))
                    fig.savefig(os.path.join("../plots/explanations", self.dataset_name, 
                                           f"{self.exp_name}_{self.job_id}", 
                                           f"{test_graph.img_id}_{test_graph.y}_{filename_suffix}.pdf"))
                plt.close()
            
            # Save adata with best config info
            adata.write(os.path.join(OUT_DATA_PATH, "adatafiles", self.dataset_name, 
                                    f"{self.exp_name}_{self.job_id}_{test_graph.img_id}_lr{best_lr}_seed{best_seed}.h5ad"))
            
            # Store summary (only best config is saved, but we keep all configs for the selected LR for reference)
            results_summary.append({
                'graph_id': test_graph.img_id,
                'best_config': best_config,
                'best_lr': best_lr,
                'lr_avg_scores': lr_avg_scores,  # Average scores for each LR
                'all_configs_for_best_lr': lr_configs[best_lr]  # All seed configs for the selected LR
            })
            
            # Print final summary for this sample
            best_seed = best_config['seed']
            print(f"\n✓ Sample {test_graph.img_id}: Selected LR={best_lr} (avg score: {lr_avg_scores[best_lr]:.4f}), "
                  f"best seed={best_seed}, score={best_score:.4f}, "
                  f"fidelity={best_config['metrics']['fidelity']:.4f}, "
                  f"sparsity={best_config['metrics']['sparsity']:.4f}")
        
        # Print overall summary of LR selection across all samples
        if len(lr_list) > 1 and len(results_summary) > 0:
            from collections import Counter
            lr_counts = Counter([r['best_config']['lr'] for r in results_summary])
            print("\n" + "="*80)
            print("Learning Rate Selection Summary Across All Samples:")
            print("="*80)
            print(f"{'Learning Rate':<15} {'Count':<10} {'Percentage':<10}")
            print("-"*80)
            total_samples = len(results_summary)
            for lr_val in sorted(lr_counts.keys()):
                count = lr_counts[lr_val]
                percentage = 100 * count / total_samples
                print(f"{lr_val:<15.4f} {count:<10} {percentage:<10.1f}%")
            print("="*80)
            print(f"Total samples: {total_samples}")
            print("="*80 + "\n")
        
        return results_summary

    def explain(self,lr: float=0.1,  epoch: int=200, return_type: str = "regression", feat_mask_type: str = "feature"):
        
        
        gene_list = custom_tools.get_gene_list(self.dataset_name)
        print(gene_list)
        
        adata_concat = []
        count = 0
        for test_graph in tqdm(self.dataset):

            explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=epoch, lr=lr),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode= return_type,
                task_level='graph',
                return_type='raw', 
            ),
            )
            # print(f"{test_graph.img_id}_{test_graph.y.item()}_{lr}")
            # print(test_graph)
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
            if np.max(edge_exp_score_mask_arr) > 0.001:

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
                        fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.y.item()}_lr-{lr}.png"))
                        fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.y.item()}_lr-{lr}.pdf"))
                    else:
                        fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.y.item()}_lr-{lr}.png"))
                        fig.savefig( os.path.join("../plots/explanations", self.dataset_name, f"{self.exp_name}_{self.job_id}", f"{test_graph.img_id}_{test_graph.y.item()}_lr-{lr}.pdf"))
                    plt.close()
                count +=1
                adata.write(os.path.join(OUT_DATA_PATH, "adatafiles", self.dataset_name, f"{self.exp_name}_{self.job_id}_{test_graph.img_id}_{test_graph.y.item()}_lr-{lr}.h5ad"))
            



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