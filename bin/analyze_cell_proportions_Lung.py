import os
import warnings
import numpy as np
import scanpy as sc
import custom_tools
import anndata as ad
import seaborn as sns
import pandas as pd
import decoupler as dc
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
from dataset import TissueDataset
from pathlib import Path
import matplotlib
from dataset import TissueDataset, LungDataset
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests


warnings.simplefilter(action='ignore')
sc.settings.verbosity = 0
# Set figure params
sc.set_figure_params(scanpy=True, facecolor="white", dpi=80, dpi_save=300)


df_merged_preprocessed_dataset = pd.read_csv("/home/rifaioglu/projects/GNNClinicalOutcomePrediction/data/Lung/raw/merged_preprocessed_dataset.csv")

ct_dict = {}
for ind, row in df_merged_preprocessed_dataset.iterrows():
    s_id = row["sample_id"]
    if f"Lung_Lung_{s_id}_{s_id}" in ct_dict.keys():
        
        ct_dict[f"Lung_Lung_{s_id}_{s_id}"].append(row["cell_type"])
    else:
        ct_dict[f"Lung_Lung_{s_id}_{s_id}"] = [row["cell_type"]]

ct_dict


# Read json file
# exp_name = "GATV2_NegativeLogLikelihood_month_04-12-2023"
# job_id = "fombnNMthdocYhDPmAjaBQ"
# exp_name = "progression_hpo_full_training"
exp_name = "Lung"
dataset_name = "Lung"
# job_id = "2mYvWJwUarIJkw0vvUbl3Q"


# job_id = "Mx08rzAp6WZs_yDoXS1Fuw"  # This is the best model trained fully on the Lung dataset named as Lung
job_id = "Lung"

PLT_PATH = f"../plots/analysis/{exp_name}_{job_id}/proportions"
Path(PLT_PATH).mkdir(parents=True, exist_ok=True)
# Create output directory
output_dir = Path(f"/home/rifaioglu/projects/GNNClinicalOutcomePrediction/plots/analysis/{exp_name}_{job_id}/proportions")
output_dir.mkdir(parents=True, exist_ok=True)
device =  custom_tools.get_device()
args  = custom_tools.load_json(f"../models/{exp_name}/{job_id}.json")


print(args)
# args["num_node_features"] = 33
deg = None
if "PNA" in exp_name:
    deg = custom_tools.load_pickle(f"../models/{exp_name}/{job_id}_deg.pckl")
model = custom_tools.load_model(f"{job_id}_SD", path = f"../models/{exp_name}", model_type = "SD", args = args, deg=deg, label_type = "classification", device=device)
dataset = LungDataset(os.path.join(f"../data/{dataset_name}"),  "Progression")

adata_concat = []
for adata_fl in os.listdir(f"/home/rifaioglu/projects/GNNClinicalOutcomePrediction/data/out_data/adatafiles/Lung"):
    if  adata_fl.endswith("lr-0.01.h5ad"):
        adata = sc.read_h5ad(f"/home/rifaioglu/projects/GNNClinicalOutcomePrediction/data/out_data/adatafiles/Lung/{adata_fl}")
        idx = adata_fl.split("_lr")[0]
        idx = idx.rsplit("_", 1)[0]
        print(idx)
        adata.obs["cell_type"] = ct_dict[idx]
        adata_concat.append(adata)
adata_exp = ad.concat(adata_concat)
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
pid_list, img_id_list, osmonth_lst,  clinical_type_lst, disease_stage_lst, progression_lst= [], [], [], [], [], []
for data in related_data:
    print(data)
    # pid_list.append(str(data.p_id[0]))
    osmonth_lst.append(data.osmonth.item())
    img_id_list.append(data.img_id[0])
    clinical_type_lst.append(data.clinical_type[0])
    disease_stage_lst.append(str(data.disease_stage.item()))
    progression_lst.append(data.y.item())

embedding_arr.shape

adata_emb = ad.AnnData(embedding_arr)
adata_emb.var_names = [f"emb_{i}" for i in range(embedding_arr.shape[1])]
adata_emb.obs_names = img_id_list
adata_emb.obs["img_id"] = img_id_list
adata_emb.obs["img_id"] = adata_emb.obs["img_id"].astype("category")
adata_emb.obs["osmonth"] = osmonth_lst
adata_emb.obs["clinical_type"] = clinical_type_lst
adata_emb.obs["disease_stage"] = disease_stage_lst
adata_emb.obs["progression"] = progression_lst  

sc.tl.pca(adata_emb, svd_solver='arpack', random_state=42)
sc.pp.neighbors(adata_emb)
sc.tl.leiden(adata_emb, key_added = "leiden", resolution=1)
sc.tl.umap(adata_emb, random_state=42)

# Check and fix invalid UMAP coordinates before plotting
if 'X_umap' in adata_emb.obsm:
    umap_coords = adata_emb.obsm['X_umap']
    # Replace any infinite or NaN values with 0
    umap_coords = np.nan_to_num(umap_coords, nan=0.0, posinf=0.0, neginf=0.0)
    adata_emb.obsm['X_umap'] = umap_coords

cmap = sns.palettes.get_colormap("tab20")
color_dict = dict()

for ind, clust_index in enumerate(adata_emb.obs["leiden"].cat.categories):
    color_dict[clust_index] = cmap.colors[ind]

rcParams['figure.figsize']=(3,3)
# UMAP coordinates should already be fixed above, but double-check
if 'X_umap' in adata_emb.obsm:
    umap_coords = adata_emb.obsm['X_umap']
    if not np.isfinite(umap_coords).all():
        print(f"Warning: Found invalid UMAP coordinates. Replacing with zeros.")
        umap_coords = np.nan_to_num(umap_coords, nan=0.0, posinf=0.0, neginf=0.0)
        adata_emb.obsm['X_umap'] = umap_coords

# Suppress scanpy warnings for this plot
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*posx and posy should be finite values.*')
    sc.pl.umap(adata_emb, color=["progression"], palette=color_dict, show=False, legend_loc = 'on data')
plt.tight_layout()
plt.savefig(os.path.join(PLT_PATH, "lung_embedding_boxplot.pdf"))
plt.close()  # Changed from plt.show() to plt.close() to avoid blocking

progression_map = (
    adata_emb.obs
    .drop_duplicates(subset='img_id')  # keep first occurrence
    .set_index('img_id')['progression']
)
adata_exp.obs['progression'] = adata_exp.obs['img_id'].map(progression_map).astype('category')


# In decoupler 2.1.2, get_pseudobulk was moved to pp.pseudobulk
# Parameters min_cells and min_counts were removed
pdata_imp_vs_unimp = dc.pp.pseudobulk(
    adata_exp,
    sample_col='img_id',
    groups_col='importance_hard',
    mode='mean',
    empty=False  # Set to False to keep empty observations/features
)

# create a new pseudobulk anndata with only important nodes
pdata_imp = pdata_imp_vs_unimp[pdata_imp_vs_unimp.obs_names.str.contains("True"),:].copy()
new_obs_names = [obs_n.split("_")[0] for obs_n in pdata_imp.obs_names]
pdata_imp.obs_names = new_obs_names

# Map progression using img_id
progression_map = (
    adata_emb.obs
    .drop_duplicates(subset='img_id')  # keep first occurrence
    .set_index('img_id')['progression']
)
pdata_imp.obs['progression'] = pdata_imp.obs_names.map(progression_map).astype('category')

# Map UMAP coordinates using img_id (optional - not critical for proportion analysis)
emb_unique = adata_emb.obs.drop_duplicates(subset='img_id')
umap_coords = []
for img_id in pdata_imp.obs_names:
    mask = emb_unique['img_id'] == img_id
    if mask.any():
        idx_in_emb = emb_unique[mask].index[0]
        # Find the position in the original adata_emb
        pos_in_emb = list(adata_emb.obs.index).index(idx_in_emb)
        umap_coords.append(adata_emb.obsm["X_umap"][pos_in_emb])
    else:
        umap_coords.append([0, 0])  # placeholder if not found
if len(umap_coords) == len(pdata_imp):
    pdata_imp.obsm["X_umap"] = np.array(umap_coords)

# Filter for important nodes only (no class filter for Lung dataset)
selection = (adata_exp.obs["importance_hard"]=="True")
df = adata_exp[selection,:].obs.copy()

# Ensure progression is categorical
df['progression'] = df['progression'].astype('category')

# Get unique progression categories
progression_categories = sorted(df['progression'].unique())
print(f"\nProgression categories found: {progression_categories}")

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
    prop_a = cell_count_a / total_a if total_a > 0 else 0
    prop_b = cell_count_b / total_b if total_b > 0 else 0
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

# Compare progression 0 vs 1
if len(progression_categories) == 2:
    prog_a, prog_b = progression_categories[0], progression_categories[1]
    print(f"\nStatistical tests for comparison between progression {prog_a} and {prog_b}:")
    
    # Get cell type distributions
    group_a = df[df['progression'] == prog_a]['cell_type']
    group_b = df[df['progression'] == prog_b]['cell_type']
    
    print(f"\nProgression {prog_a} cell type distribution:")
    print(group_a.value_counts())
    print(f"\nProgression {prog_b} cell type distribution:")
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
            'direction': f'higher in progression {prog_b}' if test_result['prop_diff'] > 0 else f'higher in progression {prog_a}'
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
    all_results[f"{prog_a}_vs_{prog_b}"] = results_df
    # Save results to CSV
    results_df.to_csv(os.path.join(PLT_PATH, f'progression{prog_a}_vs_{prog_b}_proportion_differences_results.csv'), index=False)

    # --- Visualization: Proportion Differences ---
    print(f"\nCreating differences plot with {len(results_df)} cell types...")
    fig, ax = plt.subplots(figsize=(12, 8))
    results_df_sorted = results_df.sort_values('prop_diff')
    
    # Check if we have data to plot
    if len(results_df_sorted) == 0:
        print("Warning: No data to plot for differences visualization!")
    else:
        print(f"Plotting {len(results_df_sorted)} cell types")
        print(f"Proportion differences range: {results_df_sorted['prop_diff'].min():.4f} to {results_df_sorted['prop_diff'].max():.4f}")
        
        bars = ax.barh(results_df_sorted['cell_type'], results_df_sorted['prop_diff'])
        for bar, row in zip(bars, results_df_sorted.itertuples()):
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
        
        ax.set_xlabel(f'Difference in Proportion (Progression {prog_b} - Progression {prog_a})', fontsize=12)
        ax.set_title(f'Cell Type Proportion Differences Between Progression {prog_a} and {prog_b}\nwith FDR Correction', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Set y-axis limits to ensure all bars are visible
        ax.set_ylim(-0.5, len(results_df_sorted) - 0.5)
        
        for bar, row in zip(bars, results_df_sorted.itertuples()):
            width = bar.get_width()
            label_x_pos = width + (0.01 if width >= 0 else -0.01)
            significance = '***' if row.fdr_bh < 0.001 else '**' if row.fdr_bh < 0.05 else 'ns'
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f} {significance}', 
                   va='center', fontsize=8, ha='left' if width >= 0 else 'right')
        
        ax.text(0.02, 0.02, '*** FDR < 0.001\n** FDR < 0.05\nns not significant', 
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
        
        plt.tight_layout()
        output_png = os.path.join(PLT_PATH, f'progression{prog_a}_vs_{prog_b}_differences_significance.png')
        output_pdf = os.path.join(PLT_PATH, f'progression{prog_a}_vs_{prog_b}_differences_significance.pdf')
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_pdf, bbox_inches='tight')
        plt.close()
        print(f"Saved differences plot: {output_png}")
        print(f"Saved differences plot: {output_pdf}")

    # --- Visualization: Actual Proportions ---
    plt.figure(figsize=(12, 8))
    prop_df = pd.DataFrame({
        f'Progression {prog_a}': results_df['prop_a'],
        f'Progression {prog_b}': results_df['prop_b']
    }, index=results_df['cell_type'])
    prop_df['Difference'] = prop_df[f'Progression {prog_b}'] - prop_df[f'Progression {prog_a}']
    prop_df = prop_df.sort_values('Difference')
    ax = prop_df[[f'Progression {prog_a}', f'Progression {prog_b}']].plot(kind='barh', figsize=(12, 8))
    plt.title(f'Cell Type Proportions in Progression {prog_a} and {prog_b}\nwith FDR Correction')
    plt.xlabel('Proportion')
    for i, (idx, row) in enumerate(prop_df.iterrows()):
        cell_type = idx
        fdr_bh = results_df[results_df['cell_type'] == cell_type]['fdr_bh'].iloc[0]
        significance = '***' if fdr_bh < 0.001 else '**' if fdr_bh < 0.05 else 'ns'
        max_prop = max(row[f'Progression {prog_a}'], row[f'Progression {prog_b}'])
        plt.text(max_prop + 0.01, i, significance, va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_PATH, f'progression{prog_a}_vs_{prog_b}_proportions_significance.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLT_PATH, f'progression{prog_a}_vs_{prog_b}_proportions_significance.pdf'), bbox_inches='tight')
    plt.close()

# --- Combined Visualization: Proportions Across Progression Categories ---
print("\nCreating combined visualization of proportions across progression categories...")
prop_data = {}
for prog_cat in progression_categories:
    prog_data = df[df['progression'] == prog_cat]['cell_type']
    if len(prog_data) > 0:
        prop_data[f'Progression {prog_cat}'] = pd.Series(prog_data).value_counts(normalize=True)
    else:
        print(f"Warning: No data found for progression {prog_cat}")

if len(prop_data) > 0:
    prop_df = pd.DataFrame(prop_data).fillna(0)
    # Ensure all cell types are included
    all_cell_types = sorted(df['cell_type'].unique())
    prop_df = prop_df.reindex(all_cell_types, fill_value=0)
    
    # Sort by total proportion across all categories
    prop_df['Total'] = prop_df.sum(axis=1)
    prop_df = prop_df.sort_values('Total', ascending=False)
    
    # Create a copy for saving (with cell_type as a column)
    prop_df_sorted = prop_df.drop('Total', axis=1).copy()
    prop_df_sorted = prop_df_sorted.reset_index()
    # Rename the index column to Cell_Type
    prop_df_sorted.rename(columns={prop_df_sorted.columns[0]: 'Cell_Type'}, inplace=True)
    
    # Save proportions dataframe
    proportions_output_file = os.path.join(PLT_PATH, 'cell_type_proportions_by_progression.csv')
    prop_df_sorted.to_csv(proportions_output_file, index=False)
    print(f"\nSaved proportions dataframe to: {proportions_output_file}")
    
    print(f"Proportions DataFrame shape: {prop_df.shape}")
    print(f"Proportions DataFrame columns: {prop_df.columns.tolist()}")
    print(f"Proportions DataFrame index (cell types): {prop_df.index.tolist()[:10]}...")
    print(f"Sample values:\n{prop_df.head()}")
    
    # Use prop_df (with index, without Total) for plotting
    prop_df = prop_df.drop('Total', axis=1)
    
    # Transpose for stacked bar plot: each bar is a progression category, segments are cell types
    prop_df_transposed = prop_df.T
    
    # Create stacked bar plot (vertical bars)
    fig, ax = plt.subplots(figsize=(12, 8))
    prop_df_transposed.plot(kind='bar', stacked=True, ax=ax, width=0.8)
    plt.title('Cell Type Proportions Across Progression Categories', fontsize=14)
    plt.xlabel('Progression Category', fontsize=12)
    plt.ylabel('Proportion', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_PATH, 'all_progression_proportions.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLT_PATH, 'all_progression_proportions.pdf'), bbox_inches='tight')
    plt.close()
    
    # Also create horizontal stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 10))
    prop_df_transposed.plot(kind='barh', stacked=True, ax=ax, width=0.8)
    plt.title('Cell Type Proportions Across Progression Categories', fontsize=14)
    plt.xlabel('Proportion', fontsize=12)
    plt.ylabel('Progression Category', fontsize=12)
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLT_PATH, 'all_progression_proportions_horizontal.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLT_PATH, 'all_progression_proportions_horizontal.pdf'), bbox_inches='tight')
    plt.close()
    
    print("Stacked bar plots saved successfully!")
else:
    print("Error: No proportion data found to plot!")

# Save the combined numerical results
if len(all_results) > 0:
    combined_results = pd.concat(list(all_results.values()))
    combined_results.to_csv(os.path.join(PLT_PATH, 'all_progression_comparisons_results.csv'), index=False)
    print(f"\nAnalysis complete! Results saved to {PLT_PATH}")
else:
    print("\nWarning: No comparisons were performed. Check that progression categories are properly set.")

