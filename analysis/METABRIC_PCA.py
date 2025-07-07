import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys

# Add the bin directory to sys.path so custom_tools can be imported
sys.path.append("bin")
from custom_tools import get_gene_list

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/METABRIC/raw/merged_preprocessed_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/PCA/METABRIC")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the data
print("Reading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Get marker names from custom_tools
marker_names = get_gene_list("METABRIC")

# Find all Intensity_MeanIntensity_FullStackc_* columns
intensity_cols = [col for col in df.columns if col.startswith('Intensity_MeanIntensity_FullStackc_')]

# Create mapping: intensity_col -> marker_name
col_to_marker = {}
for i, marker in enumerate(marker_names):
    if i < len(intensity_cols):
        col_to_marker[intensity_cols[i]] = marker
    else:
        print(f"No intensity column for marker {marker}")

# Group by patient and calculate mean expression for each marker
print("Calculating mean expression per patient...")
df_grouped = df.groupby('PID')[intensity_cols + ['OSmonth']].mean().reset_index()

# Prepare data for PCA
X = df_grouped[intensity_cols]
y = df_grouped['OSmonth']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
print("Performing PCA...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create a figure with two subplots
plt.figure(figsize=(15, 6))

# Plot 1: PCA scatter plot colored by OSmonth
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='OSmonth')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance explained)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance explained)')
plt.title('PCA of Patient Protein Profiles\nColored by OSmonth')

# Plot 2: Cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance')
plt.grid(True)

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_analysis.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_analysis.pdf'), bbox_inches='tight')
plt.close()

# Create a heatmap of the top 10 loadings for PC1 and PC2
n_top_genes = 10
pc1_loadings = pd.Series(pca.components_[0], index=marker_names).abs().nlargest(n_top_genes)
pc2_loadings = pd.Series(pca.components_[1], index=marker_names).abs().nlargest(n_top_genes)

# Combine the loadings
loadings_df = pd.DataFrame({
    'PC1': pc1_loadings,
    'PC2': pc2_loadings
})

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(loadings_df, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Top 10 Gene Loadings for PC1 and PC2')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_loadings_heatmap.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_loadings_heatmap.pdf'), bbox_inches='tight')
plt.close()

# Save the loadings to a CSV file
loadings_df.to_csv(os.path.join(OUTPUT_DIR, 'pca_loadings.csv'))

print("Analysis complete! Results saved in:", OUTPUT_DIR) 