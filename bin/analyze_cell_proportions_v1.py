import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import custom_tools
from dataset import TissueDataset
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations

# Create output directory
output_dir = Path("/home/rifaioglu/projects/GNNClinicalOutcomePrediction/plots/analysis/JacksonFischer_JF/proportions")
output_dir.mkdir(parents=True, exist_ok=True)

# After the data loading section and before the pairwise comparisons
# Create a directory for intermediate results
intermediate_dir = output_dir / 'intermediate_results'
intermediate_dir.mkdir(parents=True, exist_ok=True)

# Function to get the filename for intermediate results
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

# Read the data
exp_name = "JacksonFischer"
dataset_name = "JacksonFischer"
job_id = "JF"
device = custom_tools.get_device()
args = custom_tools.load_json(f"../models/{exp_name}/{job_id}.json")

# Load model and dataset
model = custom_tools.load_model(f"{job_id}_SD", path=f"../models/{exp_name}", model_type="SD", args=args, device=device)
dataset = TissueDataset(os.path.join(f"../data/{dataset_name}", "month"), "month")

# Read the explanations adata
adata_exp = sc.read_h5ad(f"../data/out_data/adatafiles/{dataset_name}/{exp_name}_{job_id}_concatenated_explanations.h5ad")
adata_exp.obs_names_make_unique()



# Get embeddings and perform clustering
import embeddings
emd, related_data = embeddings.get_intermediate_embeddings_for_dataset(model, dataset, batch_size=1)
embedding_arr = np.array(emd[0])

# Create AnnData object for embeddings
adata_emb = sc.AnnData(embedding_arr)
adata_emb.var_names = [f"emb_{i}" for i in range(embedding_arr.shape[1])]
adata_emb.obs_names = [data.img_id[0] for data in related_data]

# Perform clustering
sc.tl.pca(adata_emb, svd_solver='arpack', random_state=0)
sc.pp.neighbors(adata_emb)
sc.tl.leiden(adata_emb, key_added="leiden", resolution=0.1)

# Add leiden clusters to adata_exp
for cat in adata_emb.obs["leiden"].cat.categories:
    img_ids = adata_emb[adata_emb.obs["leiden"]==cat].obs_names
    adata_exp.obs.loc[adata_exp.obs["img_id"].isin(img_ids), "leiden"] = cat

# Apply selection criteria
selection = (adata_exp.obs["importance_hard"]=="True") & (adata_exp.obs["class"]=="Tumor")
df = adata_exp[selection,:].obs

# Get unique clusters
clusters = sorted(df['leiden'].unique())

# For each pairwise comparison
for cluster_a, cluster_b in combinations(clusters, 2):
    print(f"\nTwo-tailed permutation test for comparison between clusters {cluster_a} and {cluster_b}:")
    group_a = df[df['leiden'] == cluster_a]['cell_type'].tolist()
    group_b = df[df['leiden'] == cluster_b]['cell_type'].tolist()
    print(f"\nCluster {cluster_a} cell type distribution:")
    print(pd.Series(group_a).value_counts())
    print(f"\nCluster {cluster_b} cell type distribution:")
    print(pd.Series(group_b).value_counts())
    cell_types = set(group_a + group_b)
    results = []
    for cell_type in cell_types:
        print(f"\nProcessing {cell_type}...")
        obs_diff, perm_p = parallel_permutation_test(df, 'leiden', 'cell_type', cluster_a, cluster_b, cell_type, n_permutations=10000)
        # Calculate actual difference (not absolute) for visualization
        actual_diff = (pd.Series(group_b).value_counts(normalize=True).get(cell_type, 0) - 
                      pd.Series(group_a).value_counts(normalize=True).get(cell_type, 0))
        results.append({
            'cell_type': cell_type,
            'observed_diff': obs_diff,
            'actual_diff': actual_diff,
            'p_value': perm_p,
            'direction': f'higher in cluster {cluster_b}' if actual_diff > 0 else f'higher in cluster {cluster_a}'
        })
        print(f"Observed difference: {obs_diff:.4f}")
        print(f"Two-tailed permutation p-value: {perm_p:.4f}")
        print(f"Direction: {results[-1]['direction']}")
    # Sort results by p-value and print summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    print("\nSummary of results (sorted by p-value):")
    print(results_df.to_string(index=False))
    # Create visualization with significance indicators
    plt.figure(figsize=(12, 8))
    # Sort by actual difference for better visualization
    results_df = results_df.sort_values('actual_diff')
    # Create horizontal bar plot
    bars = plt.barh(results_df['cell_type'], results_df['actual_diff'])
    # Color bars based on direction and significance
    for bar, row in zip(bars, results_df.itertuples()):
        if row.p_value < 0.001:
            color = '#1f77b4' if row.actual_diff < 0 else '#d62728'  # Blue for cluster_a, Red for cluster_b
            alpha = 1.0
        elif row.p_value < 0.05:
            color = '#1f77b4' if row.actual_diff < 0 else '#d62728'
            alpha = 0.7
        else:
            color = 'gray'
            alpha = 0.5
        bar.set_color(color)
        bar.set_alpha(alpha)
    # Add labels and title
    plt.xlabel(f'Difference in Proportion (Cluster {cluster_b} - Cluster {cluster_a})')
    plt.title(f'Cell Type Proportion Differences Between Clusters {cluster_a} and {cluster_b}\nwith Statistical Significance')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    # Add value labels and significance indicators
    for bar, row in zip(bars, results_df.itertuples()):
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        significance = '***' if row.p_value < 0.001 else '**' if row.p_value < 0.05 else 'ns'
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f} {significance}', va='center')
    # Add legend for significance levels
    plt.text(0.02, 0.02, '*** p < 0.001\n** p < 0.05\nns not significant', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    # Save the plot
    plt.savefig(output_dir / f'cluster{cluster_a}_vs_{cluster_b}_differences_significance.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Create a second visualization showing the actual proportions
    plt.figure(figsize=(12, 8))
    # Calculate proportions for each cell type in both clusters
    group_a_props = pd.Series(group_a).value_counts(normalize=True)
    group_b_props = pd.Series(group_b).value_counts(normalize=True)
    # Create DataFrame for plotting
    prop_df = pd.DataFrame({
        f'Cluster {cluster_a}': group_a_props,
        f'Cluster {cluster_b}': group_b_props
    }).fillna(0)
    # Sort by the difference between clusters
    prop_df['Difference'] = prop_df[f'Cluster {cluster_b}'] - prop_df[f'Cluster {cluster_a}']
    prop_df = prop_df.sort_values('Difference')
    # Create grouped bar plot with significance indicators
    ax = prop_df[[f'Cluster {cluster_a}', f'Cluster {cluster_b}']].plot(kind='barh', figsize=(12, 8))
    plt.title(f'Cell Type Proportions in Clusters {cluster_a} and {cluster_b}\nwith Statistical Significance')
    plt.xlabel('Proportion')
    # Add significance indicators
    for i, (idx, row) in enumerate(prop_df.iterrows()):
        cell_type = idx
        p_value = results_df[results_df['cell_type'] == cell_type]['p_value'].iloc[0]
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.05 else 'ns'
        max_prop = max(row[f'Cluster {cluster_a}'], row[f'Cluster {cluster_b}'])
        plt.text(max_prop + 0.01, i, significance, va='center')
    plt.tight_layout()
    # Save the plot
    plt.savefig(output_dir / f'cluster{cluster_a}_vs_{cluster_b}_proportions_significance.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Save numerical results
    results_df.to_csv(output_dir / f'cluster{cluster_a}_vs_{cluster_b}_proportion_differences_results.csv', index=False)

# After all pairwise comparisons are done, create a combined visualization
print("\nCreating combined visualization of all pairwise comparisons...")

# Create a dictionary to store all results
all_results = {}
for cluster_a, cluster_b in combinations(clusters, 2):
    results_file = output_dir / f'cluster{cluster_a}_vs_{cluster_b}_proportion_differences_results.csv'
    results_df = pd.read_csv(results_file)
    all_results[f"{cluster_a}_vs_{cluster_b}"] = results_df

# Create a combined DataFrame for visualization
cell_types = sorted(set().union(*[set(df['cell_type']) for df in all_results.values()]))
comparisons = list(all_results.keys())

# Create matrices for differences and p-values
diff_matrix = pd.DataFrame(index=cell_types, columns=comparisons, dtype=float)
pval_matrix = pd.DataFrame(index=cell_types, columns=comparisons, dtype=float)

for comp, results_df in all_results.items():
    for _, row in results_df.iterrows():
        diff_matrix.loc[row['cell_type'], comp] = float(row['actual_diff'])
        pval_matrix.loc[row['cell_type'], comp] = float(row['p_value'])

# Create the combined visualization
plt.figure(figsize=(15, 10))

# Create a custom colormap
cmap = plt.cm.RdBu_r
norm = plt.Normalize(-0.15, 0.15)  # Adjust these values based on your data range

# Plot the heatmap
im = plt.imshow(diff_matrix.values, cmap=cmap, norm=norm, aspect='auto')

# Add colorbar
cbar = plt.colorbar(im)
cbar.set_label('Difference in Proportion')

# Add significance indicators
for i in range(len(cell_types)):
    for j in range(len(comparisons)):
        pval = pval_matrix.iloc[i, j]
        if pval < 0.001:
            significance = '***'
        elif pval < 0.05:
            significance = '**'
        else:
            significance = 'ns'
        plt.text(j, i, significance, ha='center', va='center', color='black')

# Customize the plot
plt.xticks(range(len(comparisons)), comparisons, rotation=45, ha='right')
plt.yticks(range(len(cell_types)), cell_types)
plt.title('Cell Type Proportion Differences Between All Clusters\nwith Statistical Significance')

# Add legend for significance levels
plt.text(0.02, 0.02, '*** p < 0.001\n** p < 0.05\nns not significant', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the combined plot
plt.savefig(output_dir / 'all_cluster_comparisons_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second combined visualization showing actual proportions
plt.figure(figsize=(15, 10))

# Calculate proportions for each cell type in all clusters
prop_data = {}
for cluster in clusters:
    cluster_data = df[df['leiden'] == cluster]['cell_type']
    prop_data[f'Cluster {cluster}'] = pd.Series(cluster_data).value_counts(normalize=True)

# Create DataFrame for plotting
prop_df = pd.DataFrame(prop_data).fillna(0)

# Sort by the sum of proportions for better visualization
prop_df['Total'] = prop_df.sum(axis=1)
prop_df = prop_df.sort_values('Total', ascending=False)
prop_df = prop_df.drop('Total', axis=1)

# Create grouped bar plot
ax = prop_df.plot(kind='barh', figsize=(15, 10))
plt.title('Cell Type Proportions Across All Clusters')
plt.xlabel('Proportion')
plt.legend(title='Cluster')

# Add significance indicators
for i, (idx, row) in enumerate(prop_df.iterrows()):
    cell_type = idx
    for comp, results_df in all_results.items():
        cluster_a, cluster_b = comp.split('_vs_')
        p_value = results_df[results_df['cell_type'] == cell_type]['p_value'].iloc[0]
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.05 else 'ns'
        max_prop = max(row[f'Cluster {cluster_a}'], row[f'Cluster {cluster_b}'])
        plt.text(max_prop + 0.01, i, significance, va='center')

plt.tight_layout()

# Save the combined proportions plot
plt.savefig(output_dir / 'all_cluster_proportions.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the combined numerical results
combined_results = pd.concat([
    pd.read_csv(output_dir / f'cluster{cluster_a}_vs_{cluster_b}_proportion_differences_results.csv')
    for cluster_a, cluster_b in combinations(clusters, 2)
])
combined_results.to_csv(output_dir / 'all_cluster_comparisons_results.csv', index=False) 