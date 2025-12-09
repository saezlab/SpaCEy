import os
import warnings
import numpy as np
import scanpy as sc
import custom_tools
import anndata as ad
import seaborn as sns
import decoupler as dc
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
from dataset import TissueDataset
from pathlib import Path
import matplotlib
from itertools import combinations
from multiprocessing import Pool, cpu_count
from scipy.stats import fisher_exact, chi2_contingency
from itertools import combinations
from statsmodels.stats.multitest import multipletests

warnings.simplefilter(action='ignore')
sc.settings.verbosity = 0
# Set figure params
sc.set_figure_params(scanpy=True, facecolor="white", dpi=80, dpi_save=300)
# Create output directory
output_dir = Path("./plots/analysis/JacksonFischer_JF/proportions")
output_dir.mkdir(parents=True, exist_ok=True)

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


exp_name = "JacksonFischer"
dataset_name = "JacksonFischer"
job_id = "JF"
PLT_PATH = f"../plots/analysis/{exp_name}_{job_id}/proportions"
Path(PLT_PATH).mkdir(parents=True, exist_ok=True)
device =  custom_tools.get_device()
args  = custom_tools.load_json(f"../models/{exp_name}/{job_id}.json")

# After the data loading section and before the pairwise comparisons
# Create a directory for intermediate results
intermediate_dir = output_dir / 'intermediate_results'
intermediate_dir.mkdir(parents=True, exist_ok=True)

# args["num_node_features"] = 33
deg = None
if "PNA" in exp_name:
    deg = custom_tools.load_pickle(f"../models/{exp_name}/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = f"../models/{exp_name}", model_type = "SD", args = args, deg=deg, device=device)
dataset = TissueDataset(os.path.join(f"../data/{dataset_name}", "month"),  "month")

# Read the explanations adata
adata_exp = sc.read_h5ad(f"../data/out_data/adatafiles/{dataset_name}/{exp_name}_{job_id}_concatenated_explanations.h5ad")
adata_exp.obs_names_make_unique()

imp_threshold = 0.75
# Get the importance of the nodes
node_importance = np.array(adata_exp.obs["importance"])
node_imp_thr = np.quantile(node_importance, imp_threshold)

importances_hard_v2 = np.array(node_importance > node_imp_thr, dtype="str")
# print("importances_hard", importances_hard)
importances_hard_v2 = pd.Series(importances_hard_v2, dtype="category")
# print(importances_hard)
adata_exp.obs["importance_hard"] = importances_hard_v2.values

import embeddings
emd, related_data = embeddings.get_intermediate_embeddings_for_dataset(model, dataset, batch_size=1)
emd_cnv, related_data_cnv = embeddings.get_intermediate_embeddings_for_dataset(model, dataset, mode="CNV", batch_size=1)

embedding_arr = np.array(emd[0])
pid_list, img_id_list, osmonth_lst,  clinical_type_lst, tumor_grade_lst, censor_lst= [], [], [], [], [], []
for data in related_data:
    pid_list.append(str(data.p_id[0]))
    osmonth_lst.append(data.osmonth.item())
    img_id_list.append(data.img_id[0])
    clinical_type_lst.append(data.clinical_type[0])
    tumor_grade_lst.append(str(data.tumor_grade.item()))
    censor_lst.append(data.is_censored[0].item())
embedding_arr.shape

adata_emb = ad.AnnData(embedding_arr)
adata_emb.var_names = [f"emb_{i}" for i in range(embedding_arr.shape[1])]
adata_emb.obs_names = img_id_list
adata_emb.obs["img_id"] = img_id_list
adata_emb.obs["img_id"] = adata_emb.obs["img_id"].astype("category")
adata_emb.obs["osmonth"] = osmonth_lst
adata_emb.obs["p_id"] = pid_list
adata_emb.obs["clinical_type"] = clinical_type_lst
adata_emb.obs["tumor_grade"] = tumor_grade_lst
adata_emb.obs["is_censored"] = censor_lst

sc.tl.pca(adata_emb, svd_solver='arpack', random_state=0)
sc.pp.neighbors(adata_emb)
sc.tl.leiden(adata_emb, key_added = "leiden", resolution=0.1)
sc.tl.umap(adata_emb)
# sc.pl.umap(adata, color=["osmonth", "leiden"])
upper_quartile = int(np.quantile(adata_emb.obs["osmonth"], 0.75))
lower_quartile = int(np.quantile(adata_emb.obs["osmonth"], 0.25))

adata_emb.obs["high_surv"]="0"
adata_emb.obs["low_surv"]="0"

adata_emb.obs.loc[adata_emb.obs["osmonth"]>upper_quartile, "high_surv" ] = "1"
adata_emb.obs.loc[adata_emb.obs["osmonth"]<lower_quartile, "low_surv" ] = "1"

def get_intermediate_filename(cluster_a, cluster_b, cell_type):
    return intermediate_dir / f'perm_test_cluster{cluster_a}_vs_{cluster_b}_{cell_type.replace("+", "plus").replace(" ", "_")}.npy'


def single_permutation(args):
    """
    Perform a single permutation test iteration.
    """
    combined, cluster_a, cluster_b, cell_type = args
    shuffled = combined['leiden'].sample(frac=1, replace=False).values
    combined['shuffled_cluster'] = shuffled
    group_a_perm = combined[combined['shuffled_cluster'] == cluster_a]['cell_type']
    group_b_perm = combined[combined['shuffled_cluster'] == cluster_b]['cell_type']
    # Two-tailed test: absolute difference
    diff = abs((group_b_perm == cell_type).mean() - (group_a_perm == cell_type).mean())
    return diff

def get_intermediate_filename(cluster_a, cluster_b, cell_type):
    return intermediate_dir / f'perm_test_cluster{cluster_a}_vs_{cluster_b}_{cell_type.replace("+", "plus").replace(" ", "_")}.npy'

def parallel_permutation_test(df, cluster_col, celltype_col, cluster_a, cluster_b, cell_type, n_permutations=10000, seed=42):
    """
    Perform two-tailed permutation test in parallel with intermediate result caching.
    """
    # Check if we have cached results
    cache_file = get_intermediate_filename(cluster_a, cluster_b, cell_type)
    if cache_file.exists():
        print(f"Loading cached results for {cell_type}...")
        cached_results = np.load(cache_file, allow_pickle=True).item()
        return cached_results['obs_diff'], cached_results['p_value']

    np.random.seed(seed)
    # Get observed difference in proportions
    group_a = df[df[cluster_col] == cluster_a][celltype_col]
    group_b = df[df[cluster_col] == cluster_b][celltype_col]
    # Two-tailed test: absolute difference
    obs_diff = abs((group_b == cell_type).mean() - (group_a == cell_type).mean())
    
    # Prepare data for parallel processing
    combined = df[[cluster_col, celltype_col]].copy()
    args = [(combined, cluster_a, cluster_b, cell_type) for _ in range(n_permutations)]
    
    # Use all available CPU cores
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    # Run permutations in parallel
    with Pool(n_cores) as pool:
        diffs = pool.map(single_permutation, args)
    
    # Calculate p-value for two-tailed test
    p_value = np.mean(np.array(diffs) >= obs_diff)
    
    # Save results
    results = {
        'obs_diff': obs_diff,
        'p_value': p_value,
        'diffs': diffs
    }
    np.save(cache_file, results)
    
    return obs_diff, p_value

def compare_proportions(population1, population2, cell_type='T-cell', use_fisher=False):
    """
    Compares the proportions of a specific cell type in two populations.
    """
    # Count occurrences of the cell type in both populations
    t_cell_count1 = population1.count(cell_type)
    t_cell_count2 = population2.count(cell_type)
    
    # Get the total population sizes
    total1 = len(population1)
    total2 = len(population2)
    
    # Create a 2x2 contingency table
    contingency_table = [
        [t_cell_count1, total1 - t_cell_count1],
        [t_cell_count2, total2 - t_cell_count2]
    ]
    
    if use_fisher:
        _, p_value = fisher_exact(contingency_table)
        stat_test = "Fisher Exact"
    else:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        stat_test = "Chi-square"

    return p_value, stat_test


pdata_imp_vs_unimp = dc.get_pseudobulk(
    adata_exp,
    sample_col='img_id',
    groups_col='importance_hard',
    mode='mean',
    min_cells=0,
    min_counts=0
)

# create a new pseudobulk anndata with only important nodes
pdata_imp = pdata_imp_vs_unimp[pdata_imp_vs_unimp.obs_names.str.contains("True"),:].copy()
new_obs_names = [obs_n.split("_")[0] for obs_n in pdata_imp.obs_names]
pdata_imp.obs_names = new_obs_names

pdata_imp.obs["leiden"] = adata_emb.obs["leiden"]
pdata_imp.obsm["X_umap"] = adata_emb.obsm["X_umap"]


# TODO: Add leiden clusters to adata_exp

for cat in adata_emb.obs["leiden"].cat.categories:
    img_ids = adata_emb[adata_emb.obs["leiden"]==cat].obs["img_id"].cat.categories
    adata_exp.obs.loc[adata_exp.obs["img_id"].isin(img_ids), "leiden"] = cat

adata_exp.obs["leiden"] = adata_exp.obs["leiden"].astype("category")

selection = (adata_exp.obs["importance_hard"]=="True") & (adata_exp.obs["class"]=="Tumor")
df =  adata_exp[selection,:].obs.copy()

# Get unique clusters
clusters = sorted(df['leiden'].unique())


def perform_statistical_test(group_a, group_b, cell_type, use_fisher=False):
    """
    Perform Chi-square or Fisher's exact test for a specific cell type between two groups.
    Returns p-value, test statistic, and the contingency table.
    """
    # Count occurrences of the cell type in both groups
    cell_count_a = (group_a == cell_type).sum()
    cell_count_b = (group_b == cell_type).sum()
    
    # Get the total population sizes
    total_a = len(group_a)
    total_b = len(group_b)
    
    # Create a 2x2 contingency table
    contingency_table = np.array([
        [cell_count_a, total_a - cell_count_a],
        [cell_count_b, total_b - cell_count_b]
    ])
    
    if use_fisher:
        # Use Fisher's exact test for small sample sizes
        stat, p_value = fisher_exact(contingency_table)
        test_name = "Fisher's exact test"
    else:
        # Use Chi-square test for larger sample sizes
        stat, p_value, _, _ = chi2_contingency(contingency_table)
        test_name = "Chi-square test"
    
    # Calculate proportion difference
    prop_a = cell_count_a / total_a
    prop_b = cell_count_b / total_b
    diff = prop_b - prop_a
    
    return {
        'p_value': p_value,
        'statistic': stat,
        'test_name': test_name,
        'contingency_table': contingency_table,
        'prop_diff': diff,
        'prop_a': prop_a,
        'prop_b': prop_b
    }


# Store all results
all_results = {}

# For each pairwise comparison
for cluster_a, cluster_b in combinations(clusters, 2):
    print(f"\nStatistical tests for comparison between clusters {cluster_a} and {cluster_b}:")
    
    # Get cell type distributions
    group_a = df[df['leiden'] == cluster_a]['cell_type']
    group_b = df[df['leiden'] == cluster_b]['cell_type']
    
    print(f"\nCluster {cluster_a} cell type distribution:")
    print(group_a.value_counts())
    print(f"\nCluster {cluster_b} cell type distribution:")
    print(group_b.value_counts())
    
    # Get unique cell types
    cell_types = sorted(set(group_a.unique()) | set(group_b.unique()))
    
    # Perform tests for each cell type
    results = []
    for cell_type in cell_types:
        print(f"\nProcessing {cell_type}...")
        
        # Determine which test to use based on sample size
        min_count = min((group_a == cell_type).sum(), (group_b == cell_type).sum())
        use_fisher = min_count < 5  # Use Fisher's exact test for small counts
        
        # Perform the test
        test_result = perform_statistical_test(group_a, group_b, cell_type, use_fisher=use_fisher)
        
        results.append({
            'cell_type': cell_type,
            'p_value': test_result['p_value'],
            'statistic': test_result['statistic'],
            'test_name': test_result['test_name'],
            'prop_diff': test_result['prop_diff'],
            'prop_a': test_result['prop_a'],
            'prop_b': test_result['prop_b'],
            'direction': f'higher in cluster {cluster_b}' if test_result['prop_diff'] > 0 else f'higher in cluster {cluster_a}'
        })
        
        print(f"Test: {test_result['test_name']}")
        print(f"P-value: {test_result['p_value']:.4f}")
        print(f"Proportion difference: {test_result['prop_diff']:.4f}")
        print(f"Direction: {results[-1]['direction']}")
    
    # Convert results to DataFrame and sort by p-value
    results_df = pd.DataFrame(results)
    # FDR correction
    reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['fdr_bh'] = pvals_corrected
    results_df = results_df.sort_values('fdr_bh')
    
    print("\nSummary of results (sorted by p-value):")
    print(results_df.to_string(index=False))
    
    # Store results
    all_results[f"{cluster_a}_vs_{cluster_b}"] = results_df
    # Save results to CSV
    results_df.to_csv(os.path.join(PLT_PATH, f'cluster{cluster_a}_vs_{cluster_b}_proportion_differences_results.csv'), index=False)

    # --- Visualization: Proportion Differences ---
    plt.figure(figsize=(12, 8))
    results_df = results_df.sort_values('prop_diff')
    bars = plt.barh(results_df['cell_type'], results_df['prop_diff'])
    for bar, row in zip(bars, results_df.itertuples()):
        if row.fdr_bh < 0.001:
            color = '#1f77b4' if row.prop_diff < 0 else '#d62728'
            alpha = 1.0
        elif row.fdr_bh < 0.05:
            color = '#1f77b4' if row.prop_diff < 0 else '#d62728'
            alpha = 0.7
        else:
            color = 'gray'
            alpha = 0.5
        bar.set_color(color)
        bar.set_alpha(alpha)
    plt.xlabel(f'Difference in Proportion (Cluster {cluster_b} - Cluster {cluster_a})')
    plt.title(f'Cell Type Proportion Differences Between Clusters {cluster_a} and {cluster_b}\nwith FDR Correction')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    for bar, row in zip(bars, results_df.itertuples()):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        significance = '***' if row.fdr_bh < 0.001 else '**' if row.fdr_bh < 0.05 else 'ns'
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f} {significance}', va='center')
    plt.text(0.02, 0.02, '*** FDR < 0.001\n** FDR < 0.05\nns not significant', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_PATH, f'cluster{cluster_a}_vs_{cluster_b}_differences_significance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLT_PATH, f'cluster{cluster_a}_vs_{cluster_b}_differences_significance.pdf'), bbox_inches='tight')
    plt.close()

    # --- Visualization: Actual Proportions ---
    plt.figure(figsize=(12, 8))
    prop_df = pd.DataFrame({
        f'Cluster {cluster_a}': results_df['prop_a'],
        f'Cluster {cluster_b}': results_df['prop_b']
    }, index=results_df['cell_type'])
    prop_df['Difference'] = prop_df[f'Cluster {cluster_b}'] - prop_df[f'Cluster {cluster_a}']
    prop_df = prop_df.sort_values('Difference')
    ax = prop_df[[f'Cluster {cluster_a}', f'Cluster {cluster_b}']].plot(kind='barh', figsize=(12, 8))
    plt.title(f'Cell Type Proportions in Clusters {cluster_a} and {cluster_b}\nwith FDR Correction')
    plt.xlabel('Proportion')
    for i, (idx, row) in enumerate(prop_df.iterrows()):
        cell_type = idx
        fdr_bh = results_df[results_df['cell_type'] == cell_type]['fdr_bh'].iloc[0]
        significance = '***' if fdr_bh < 0.001 else '**' if fdr_bh < 0.05 else 'ns'
        max_prop = max(row[f'Cluster {cluster_a}'], row[f'Cluster {cluster_b}'])
        plt.text(max_prop + 0.01, i, significance, va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_PATH, f'cluster{cluster_a}_vs_{cluster_b}_proportions_significance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLT_PATH, f'cluster{cluster_a}_vs_{cluster_b}_proportions_significance.pdf'), bbox_inches='tight')
    plt.close()

# --- Combined Visualizations ---
print("\nCreating combined visualization of all pairwise comparisons...")
cell_types = sorted(set().union(*[set(df['cell_type']) for df in all_results.values()]))
comparisons = list(all_results.keys())
diff_matrix = pd.DataFrame(index=cell_types, columns=comparisons, dtype=float)
pval_matrix = pd.DataFrame(index=cell_types, columns=comparisons, dtype=float)
for comp, results_df in all_results.items():
    for _, row in results_df.iterrows():
        diff_matrix.loc[row['cell_type'], comp] = float(row['prop_diff'])
        pval_matrix.loc[row['cell_type'], comp] = float(row['fdr_bh'])
plt.figure(figsize=(15, 10))
cmap = plt.cm.RdBu_r
norm = plt.Normalize(-0.15, 0.15)
im = plt.imshow(diff_matrix.values, cmap=cmap, norm=norm, aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Difference in Proportion')
for i in range(len(cell_types)):
    for j in range(len(comparisons)):
        fdr_bh = pval_matrix.iloc[i, j]
        if fdr_bh < 0.001:
            significance = '***'
        elif fdr_bh < 0.05:
            significance = '**'
        else:
            significance = 'ns'
        plt.text(j, i, significance, ha='center', va='center', color='black')
plt.xticks(range(len(comparisons)), comparisons, rotation=45, ha='right')
plt.yticks(range(len(cell_types)), cell_types)
plt.title('Cell Type Proportion Differences Between All Clusters\nwith FDR Correction')
plt.text(0.02, 0.02, '*** FDR < 0.001\n** FDR < 0.05\nns not significant', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'all_cluster_comparisons_heatmap.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(PLT_PATH, 'all_cluster_comparisons_heatmap.pdf'), bbox_inches='tight')
plt.close()
plt.figure(figsize=(15, 10))
prop_data = {}
for cluster in clusters:
    cluster_data = df[df['leiden'] == cluster]['cell_type']
    prop_data[f'Cluster {cluster}'] = pd.Series(cluster_data).value_counts(normalize=True)
prop_df = pd.DataFrame(prop_data).fillna(0)
prop_df['Total'] = prop_df.sum(axis=1)
prop_df = prop_df.sort_values('Total', ascending=False)
prop_df = prop_df.drop('Total', axis=1)
ax = prop_df.plot(kind='barh', figsize=(15, 10))
plt.title('Cell Type Proportions Across All Clusters')
plt.xlabel('Proportion')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, 'all_cluster_proportions.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(PLT_PATH, 'all_cluster_proportions.pdf'), bbox_inches='tight')
plt.close()
# Save the combined numerical results
combined_results = pd.concat([
    pd.read_csv(os.path.join(PLT_PATH, f'cluster{cluster_a}_vs_{cluster_b}_proportion_differences_results.csv'))
    for cluster_a, cluster_b in combinations(clusters, 2)
])
combined_results.to_csv(os.path.join(PLT_PATH, 'all_cluster_comparisons_results.csv'), index=False)


