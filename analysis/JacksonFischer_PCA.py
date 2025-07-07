import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys

# Add the bin directory to the Python path
sys.path.append("/home/rifaioglu/projects/GNNClinicalOutcomePrediction/bin")
from data_preparation import get_basel_zurich_staining_panel

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/JacksonFischer/raw/merged_preprocessed_dataset.csv")
STAINING_PANEL_PATH = os.path.join(BASE_DIR, "data/JacksonFischer/Basel_Zuri_StainingPanel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/PCA/JacksonFischer")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the data
print("Reading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Get protein columns (Intensity_Mean columns)
protein_cols = [col for col in df.columns if 'Intensity_Mean' in col]

# Get protein mapping
print("Reading staining panel...")
staining_panel_df = pd.read_csv(STAINING_PANEL_PATH)
protein_mapping = {}
for _, row in staining_panel_df.iterrows():
    target = row["Target"]
    full_stack = row["FullStack"]
    if full_stack not in [1,2,3,4,5,6,7,8,26,32,36,42,48,49]:  # unwanted_ids
        protein_mapping[full_stack] = target

# Create a mapping from column names to protein names
col_to_protein = {}
for col in protein_cols:
    try:
        channel_id = int(col.split("Intensity_MeanIntensity_FullStack_c")[1])
        if channel_id in protein_mapping:
            col_to_protein[col] = protein_mapping[channel_id]
    except:
        continue

# Group by patient and calculate mean expression for each marker
print("Calculating mean expression per patient...")
df_grouped = df.groupby('PID')[protein_cols + ['OSmonth']].mean().reset_index()

# Prepare data for PCA
X = df_grouped[protein_cols]
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
n_top_proteins = 10
protein_names = [col_to_protein[col] for col in protein_cols]
pc1_loadings = pd.Series(pca.components_[0], index=protein_names).abs().nlargest(n_top_proteins)
pc2_loadings = pd.Series(pca.components_[1], index=protein_names).abs().nlargest(n_top_proteins)

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