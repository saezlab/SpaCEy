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
import pandas as pd
import plotting
import matplotlib
from itertools import combinations
from multiprocessing import Pool, cpu_count
from scipy.stats import fisher_exact, chi2_contingency
from itertools import combinations
from statsmodels.stats.multitest import multipletests


# Read json file
# exp_name = "GATV2_NegativeLogLikelihood_month_04-12-2023"
# job_id = "fombnNMthdocYhDPmAjaBQ"
exp_name = "METABRIC"
dataset_name = "METABRIC"
job_id = "METABRIC"
PLT_PATH = f"../plots/analysis/{exp_name}_{job_id}/proportions"
Path(PLT_PATH).mkdir(parents=True, exist_ok=True)
device =  custom_tools.get_device()
args  = custom_tools.load_json(f"../models/{exp_name}/{job_id}.json")

plotting.visualize_clinical_data(plt_path=PLT_PATH)

# args["num_node_features"] = 33
deg = None
if "PNA" in exp_name:
    deg = custom_tools.load_pickle(f"../models/{exp_name}/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = f"../models/{exp_name}", model_type = "SD", args = args, deg=deg, device=device)
dataset = TissueDataset(os.path.join(f"../data/{dataset_name}", "month"),  "month")

# Read the explanations adata
adata_exp = sc.read_h5ad(f"../data/out_data/adatafiles/{dataset_name}/{exp_name}_{job_id}_concatenated_explanations.h5ad")
adata_exp.obs_names_make_unique()

def classify_breast_cancer(row):
    er = row['ER Status']
    pr = row['PR Status']
    her2 = row['HER2 Status']
    
    hr_positive = (er == 'Positive') or (pr == 'Positive')

    if not hr_positive and her2 == 'Negative':
        return 'TripleNeg'
    elif not hr_positive and her2 == 'Positive':
        return 'HR-HER2+'
    elif hr_positive and her2 == 'Negative':
        return 'HR+HER2-'
    elif hr_positive and her2 == 'Positive':
        return 'HR+HER2+'
    else:
        return np.nan  # In case of unexpected values

c_data  = pd.read_csv("../data/METABRIC/brca_metabric_clinical_data.tsv", sep="\t", index_col=False)
s_c_data = pd.read_csv("../data/METABRIC/single_cell_data.csv", index_col=False)
c_data.columns = c_data.columns.str.strip()
c_data['Subtype'] = c_data.apply(classify_breast_cancer, axis=1)
# Print columns
print("Clinical data columns: ", c_data.columns)
print("Single cell data columns: ", s_c_data.columns)

# Keep rows in c_data with PIDs in single_cell_data
c_data = c_data[c_data["Patient ID"].isin(s_c_data["metabricId"])]

# Define custom order
custom_order = ["TripleNeg", "HR-HER2+", "HR+HER2-", "HR+HER2+"]

# Define darker shades
my_pal = {"TripleNeg": "#4682B4",  # Steel Blue (Darker Blue)
          "HR-HER2+": "#DAA520",   # Goldenrod (Darker Yellow)
          "HR+HER2-": "#228B22",   # Forest Green (Darker Green)
          "HR+HER2+": "#B22222"}   # Firebrick (Darker Red)


c_data  = pd.read_csv("../data/METABRIC/brca_metabric_clinical_data.tsv", sep="\t", index_col=False)

analysis_vars  = ['CK19', 'CK8_18', 'CD68', 'SMA',
       'Vimentin', 'HER2', 'CD3', 'Slug', 'ER',
       'PR', 'CD45', 'GATA3', 'CD20', 'Beta_catenin',
       'CAIX', 'Ki67', 'EGFR', 'CK7', 'panCK',
       'CK5', 'Fibronectin']

adata_exp = adata_exp[:, analysis_vars]

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
    # print(data.tumor_grade)
    tumor_grade_lst.append(str(data.tumor_grade[0]))
    # print(data.is_censored)
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
new_res_param = 0.1
sc.tl.leiden(adata_emb, restrict_to=('leiden', ["1"]),  resolution=new_res_param, key_added='leiden_clust')

adata_emb.obs['leiden_clust'][adata_emb.obs['leiden_clust'].isin(['1,1', '1,0', '2'])]='1,0'
adata_emb.obs['leiden_clust']=adata_emb.obs['leiden_clust'].astype('str').astype('category')
### Reorder and rename the Leiden
adata_emb.obs['leiden_clust'].cat.rename_categories(np.arange(len(np.unique(adata_emb.obs['leiden_clust']))).astype('str')) #, inplace=True)

new_res_param = 0.1
sc.tl.leiden(adata_emb, restrict_to=('leiden_clust', ["0"]),  resolution=new_res_param, key_added='leiden_clust')

adata_emb.obs['leiden_clust'][adata_emb.obs['leiden_clust'].isin(['1,1', '1,0', '2'])]='1,0'
### Reorder and rename the Leiden

adata_emb.obs['leiden_clust'] = adata_emb.obs['leiden_clust'].cat.rename_categories(np.arange(len(np.unique(adata_emb.obs['leiden_clust']))).astype('str')) #, inplace=True)

adata_emb.obs['leiden_clust'][adata_emb.obs['leiden_clust'].isin(['0', '1'])]='1'
adata_emb.obs['leiden_clust']=adata_emb.obs['leiden_clust'].astype('str').astype('category')
### Reorder and rename the Leiden
adata_emb.obs['leiden_clust'] = adata_emb.obs['leiden_clust'].cat.rename_categories(np.arange(len(np.unique(adata_emb.obs['leiden_clust']))).astype('str')) #, inplace=True)s


# subclustering 1
sc.tl.leiden(adata_emb, restrict_to=('leiden_clust', ["1"]),  resolution=new_res_param, key_added='leiden_clust')
adata_emb.obs['leiden_clust']=adata_emb.obs['leiden_clust'].astype('str').astype('category')
### Reorder and rename the Leiden
adata_emb.obs['leiden_clust'] = adata_emb.obs['leiden_clust'].cat.rename_categories(np.arange(len(np.unique(adata_emb.obs['leiden_clust']))).astype('str'))

# subclustering 1
sc.tl.leiden(adata_emb, restrict_to=('leiden_clust', ["3"]),  resolution=new_res_param, key_added='leiden_clust')
adata_emb.obs['leiden_clust']=adata_emb.obs['leiden_clust'].astype('str').astype('category')
### Reorder and rename the Leiden
adata_emb.obs['leiden_clust'] = adata_emb.obs['leiden_clust'].cat.rename_categories(np.arange(len(np.unique(adata_emb.obs['leiden_clust']))).astype('str'))

adata_emb.obs['leiden_clust'][adata_emb.obs['leiden_clust'].isin(['3', '5'])]='3'
adata_emb.obs['leiden_clust']=adata_emb.obs['leiden_clust'].astype('str').astype('category')
adata_emb.obs['leiden_clust'] = adata_emb.obs['leiden_clust'].cat.rename_categories(np.arange(len(np.unique(adata_emb.obs['leiden_clust']))).astype('str'))


adata_emb.obs['leiden_clust'][adata_emb.obs['leiden_clust'].isin(['4', '2', '1'])]='1'
adata_emb.obs['leiden_clust']=adata_emb.obs['leiden_clust'].astype('str').astype('category')
adata_emb.obs['leiden_clust'] = adata_emb.obs['leiden_clust'].cat.rename_categories(np.arange(len(np.unique(adata_emb.obs['leiden_clust']))).astype('str'))

annotation_dict = {"0":"2", "1":"1", "2":"0"}
adata_emb.obs['leiden_clust'] = adata_emb.obs['leiden_clust'].map(annotation_dict)

sc.tl.umap(adata_emb)
# sc.pl.umap(adata, color=["osmonth", "leiden"])
upper_quartile = int(np.quantile(adata_emb.obs["osmonth"], 0.75))
lower_quartile = int(np.quantile(adata_emb.obs["osmonth"], 0.25))

adata_emb.obs["high_surv"]="0"
adata_emb.obs["low_surv"]="0"

adata_emb.obs.loc[adata_emb.obs["osmonth"]>upper_quartile, "high_surv" ] = "1"
adata_emb.obs.loc[adata_emb.obs["osmonth"]<lower_quartile, "low_surv" ] = "1"


import decoupler as dc
pdata_imp_vs_unimp = dc.get_pseudobulk(
    adata_exp,
    
    sample_col='img_id',
    groups_col='importance_hard',
    mode='mean',
    min_cells=0,
    min_counts=0
)

pdata_imp_vs_unimp.obs_names
pdata_imp = pdata_imp_vs_unimp[pdata_imp_vs_unimp.obs_names.str.contains("True"),:].copy()
new_obs_names = [obs_n.split("_")[0] for obs_n in pdata_imp.obs_names]
pdata_imp.obs_names = new_obs_names

pdata_imp.obs["leiden_clust"] = adata_emb.obs["leiden_clust"]
pdata_imp.obsm["X_umap"] = adata_emb.obsm["X_umap"]


# TODO: Add leiden clusters to adata_exp

for cat in adata_emb.obs["leiden_clust"].cat.categories:
    img_ids = adata_emb[adata_emb.obs["leiden_clust"]==cat].obs["img_id"].cat.categories
    adata_exp.obs.loc[adata_exp.obs["img_id"].isin(img_ids), "leiden_clust"] = cat

# print(sorted(adata_emb[adata_emb.obs["leiden"]=="0"].obs["img_id"].cat.categories))

adata_exp.obs["leiden_clust"] = adata_exp.obs["leiden_clust"].astype("category")



df = adata_exp.obs.copy()
# Get unique clusters
clusters = sorted(df['leiden_clust'].unique())


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
    group_a = df[df['leiden_clust'] == cluster_a]['cell_type']
    group_b = df[df['leiden_clust'] == cluster_b]['cell_type']
    
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
    cluster_data = df[df['leiden_clust'] == cluster]['cell_type']
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


