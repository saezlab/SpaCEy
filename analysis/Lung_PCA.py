import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/Lung/raw/merged_preprocessed_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/PCA/Lung")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the data
print("Reading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Get marker/protein columns (these are the protein expression markers)
marker_cols = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", 
               "CD68", "CD8a", "CD94", "DNA1", "FoxP3", "HLA-DR", "Histone H3", "MPO", "Pancytokeratin"]

# Check if TTF1 exists and add it if present
if "TTF1" in df.columns:
    marker_cols.append("TTF1")

# Filter to only include markers that exist in the dataset
marker_cols = [col for col in marker_cols if col in df.columns]

print(f"Found {len(marker_cols)} marker columns: {marker_cols}")

# Group by patient and calculate mean expression for each marker
print("Calculating mean expression per patient...")
df_grouped = df.groupby('sample_id')[marker_cols + ['Survival or loss (years)']].mean().reset_index()

# Remove rows with missing survival data
df_grouped = df_grouped.dropna(subset=['Survival or loss (years)'])

print(f"Number of patients: {len(df_grouped)}")

# Prepare data for PCA
X = df_grouped[marker_cols]
y = df_grouped['Survival or loss (years)']

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

# Plot 1: PCA scatter plot colored by Survival
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Survival or loss (years)')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance explained)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance explained)')
plt.title('PCA of Patient Protein Profiles\nColored by Survival (years)')

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
n_top_proteins = 10
pc1_loadings = pd.Series(pca.components_[0], index=marker_cols).abs().nlargest(n_top_proteins)
pc2_loadings = pd.Series(pca.components_[1], index=marker_cols).abs().nlargest(n_top_proteins)

# Combine the loadings
loadings_df = pd.DataFrame({
    'PC1': pc1_loadings,
    'PC2': pc2_loadings
})

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(loadings_df, annot=True, cmap='YlOrRd', fmt='.3f')
plt.title('Top 10 Protein Loadings for PC1 and PC2')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_loadings_heatmap.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_loadings_heatmap.pdf'), bbox_inches='tight')
plt.close()

# Save the loadings to a CSV file
loadings_df.to_csv(os.path.join(OUTPUT_DIR, 'pca_loadings.csv'))

print("Analysis complete! Results saved in:", OUTPUT_DIR)





