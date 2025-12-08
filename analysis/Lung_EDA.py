import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/Lung/raw/merged_preprocessed_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/EDA/Lung")

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

def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)

# Get clinical columns
clinical_cols_to_group = ['Sex', 'Age', 'BMI', 'Smoking Status', 'Pack Years', 'Stage', 
                          'Progression', 'Death', 'Survival or loss (years)', 'Predominant histological pattern']
clinical_cols_to_group = [col for col in clinical_cols_to_group if col in df.columns]

# Group by patient and calculate mean expression for each marker
print("Calculating mean expression per patient...")
# For numeric columns, use mean; for categorical, use first (they should be the same per patient)
grouped_dict = {}
for col in marker_cols + clinical_cols_to_group:
    if col in df.columns:
        # Check if column is numeric (including object columns that can be converted)
        try:
            pd.to_numeric(df[col].dropna())
            grouped_dict[col] = 'mean'
        except (ValueError, TypeError):
            grouped_dict[col] = 'first'

df_grouped = df.groupby('sample_id').agg(grouped_dict).reset_index()

# Remove rows with missing survival data
df_grouped = df_grouped.dropna(subset=['Survival or loss (years)'])

print(f"Number of patients: {len(df_grouped)}")

# Split markers into two groups
half = len(marker_cols) // 2
group1 = marker_cols[:half]
group2 = marker_cols[half:]

# Create a single figure with subplots for each marker in group 1
print("Creating merged box plot figure for group 1...")
n_markers1 = len(group1)
n_cols = 6
n_rows = (n_markers1 + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 25))  # Increased height for longer box plots
axes = axes.flatten()

for i, marker in enumerate(group1):
    sns.boxplot(y=df_grouped[marker], ax=axes[i], width=0.5, color='skyblue')
    axes[i].set_title(f'Mean {marker} Expression')
    axes[i].set_ylabel('Mean Expression Level')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path_png = os.path.join(OUTPUT_DIR, 'merged_mean_boxplot_group1.png')
output_path_pdf = os.path.join(OUTPUT_DIR, 'merged_mean_boxplot_group1.pdf')
plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
plt.savefig(output_path_pdf, bbox_inches='tight')
plt.close()

# Create a single figure with subplots for each marker in group 2
print("Creating merged box plot figure for group 2...")
n_markers2 = len(group2)
n_cols = 5
n_rows = (n_markers2 + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 25))  # Increased height for longer box plots
axes = axes.flatten()

for i, marker in enumerate(group2):
    sns.boxplot(y=df_grouped[marker], ax=axes[i], width=0.5, color='skyblue')
    axes[i].set_title(f'Mean {marker} Expression')
    axes[i].set_ylabel('Mean Expression Level')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path_png = os.path.join(OUTPUT_DIR, 'merged_mean_boxplot_group2.png')
output_path_pdf = os.path.join(OUTPUT_DIR, 'merged_mean_boxplot_group2.pdf')
plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
plt.savefig(output_path_pdf, bbox_inches='tight')
plt.close()

# Perform EDA for each clinical variable after grouping by patient ID
print("Performing EDA for each clinical variable after grouping by patient ID...")
excluded_cols = marker_cols + ['sample_id', 'cell_id', 'cell_type', 'Location_Center_X', 'Location_Center_Y', 'area_pixels']
clinical_cols = [col for col in df.columns if col not in excluded_cols]
for col in clinical_cols:
    if col in df_grouped.columns:
        print(f"Analyzing {col}...")
        
        # Calculate summary statistics
        summary_stats = df_grouped[col].describe()
        summary_stats.to_csv(os.path.join(OUTPUT_DIR, f'{sanitize_filename(col)}_summary_statistics.csv'))
        
        # Create box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=df_grouped[col], color='skyblue')
        plt.title(f'Mean {col} per Patient')
        plt.ylabel('Mean Value')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{sanitize_filename(col)}_boxplot.png'), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{sanitize_filename(col)}_boxplot.pdf'))
        plt.close()
    else:
        print(f"Column {col} not found in grouped data.")

# Create summary statistics
print("Calculating summary statistics...")
summary_stats = df[marker_cols].describe()
summary_stats.to_csv(os.path.join(OUTPUT_DIR, 'protein_summary_statistics.csv'))

# Create correlation heatmap
print("Creating correlation heatmap...")
plt.figure(figsize=(15, 12))
correlation_matrix = df[marker_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Protein Expression Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'protein_correlation_heatmap.png'), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'protein_correlation_heatmap.pdf'))
plt.close()

# Perform EDA for clinical variables: box plots of numeric variables grouped by categorical variables
print("Performing EDA for clinical variables (numeric vs categorical)...")

# Define clinical variables for Lung dataset
categorical_vars = [
    'Sex', 'Age', 'BMI', 'Smoking Status', 'Pack Years', 'Stage', 
    'Progression', 'Death', 'Predominant histological pattern'
]
numeric_vars = [
    'Survival or loss (years)'
]

# Only keep those that are present in the DataFrame
categorical_vars = [col for col in categorical_vars if col in df.columns]
numeric_vars = [col for col in numeric_vars if col in df.columns]

for cat_var in categorical_vars:
    for num_var in numeric_vars:
        if cat_var in df.columns and num_var in df.columns:
            # Drop rows with missing values for this pair
            plot_df = df[[cat_var, num_var]].dropna()
            if plot_df[cat_var].nunique() < 2:
                print(f"Skipping {num_var} by {cat_var}: less than 2 groups with data.")
                continue
            print(f"Boxplot: {num_var} by {cat_var}")
            plt.figure(figsize=(10, 6))
            
            sns.boxplot(x=plot_df[cat_var], y=plot_df[num_var], color='skyblue')
            plt.title(f'{num_var} by {cat_var}')
            plt.xlabel(cat_var)
            plt.ylabel(num_var)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path_png = os.path.join(OUTPUT_DIR, f'{sanitize_filename(num_var)}_by_{sanitize_filename(cat_var)}_boxplot.png')
            plot_path_pdf = os.path.join(OUTPUT_DIR, f'{sanitize_filename(num_var)}_by_{sanitize_filename(cat_var)}_boxplot.pdf')
            plt.savefig(plot_path_png, dpi=300)
            plt.savefig(plot_path_pdf)
            plt.close()
            # Summary statistics per category
            stats = plot_df.groupby(cat_var)[num_var].describe()
            stats.to_csv(os.path.join(OUTPUT_DIR, f'{sanitize_filename(num_var)}_by_{sanitize_filename(cat_var)}_summary.csv'))

# Add scatter plots for each marker vs Survival
print("Creating scatter plots for each marker vs Survival...")

# Calculate number of rows and columns for the grid
n_markers = len(marker_cols)
n_cols = 6  # You can adjust this number to change the layout
n_rows = (n_markers + n_cols - 1) // n_cols

# Create a single figure for all scatter plots
plt.figure(figsize=(30, 5*n_rows))
for i, marker in enumerate(marker_cols):
    if marker in df_grouped.columns and 'Survival or loss (years)' in df_grouped.columns:
        print(f"Creating scatter plot for {marker} vs Survival")
        
        # Calculate correlation coefficient
        correlation = df_grouped[marker].corr(df_grouped['Survival or loss (years)'])
        
        # Create subplot
        plt.subplot(n_rows, n_cols, i+1)
        sns.regplot(x=df_grouped['Survival or loss (years)'], y=df_grouped[marker], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'{marker} (r = {correlation:.3f})')
        plt.xlabel('Survival or loss (years)')
        plt.ylabel(f'Mean {marker} Expression')
        plt.tight_layout()

# Save the merged figure
plot_path_png = os.path.join(OUTPUT_DIR, 'merged_marker_vs_Survival_scatter.png')
plot_path_pdf = os.path.join(OUTPUT_DIR, 'merged_marker_vs_Survival_scatter.pdf')
plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
plt.savefig(plot_path_pdf, bbox_inches='tight')
plt.close()

print("Analysis complete! Results saved in:", OUTPUT_DIR)

