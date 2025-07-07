import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re

# Add the bin directory to the Python path
sys.path.append("/home/rifaioglu/projects/GNNClinicalOutcomePrediction/bin")
from data_preparation import get_basel_zurich_staining_panel

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/JacksonFischer/raw/merged_preprocessed_dataset.csv")
STAINING_PANEL_PATH = os.path.join(BASE_DIR, "data/JacksonFischer/Basel_Zuri_StainingPanel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/EDA/JacksonFischer")

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

def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)

# Group by patient and calculate mean expression for each marker
print("Calculating mean expression per patient...")
df_grouped = df.groupby('PID')[protein_cols + ['OSmonth']].mean().reset_index()

# Split markers into two groups
markers = list(col_to_protein.items())
half = len(markers) // 2
group1 = markers[:half]
group2 = markers[half:]

# Create a single figure with subplots for each marker in group 1
print("Creating merged box plot figure for group 1...")
n_markers1 = len(group1)
n_cols = 6
n_rows = (n_markers1 + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 25))  # Increased height for longer box plots
axes = axes.flatten()

for i, (col, protein_name) in enumerate(group1):
    sns.boxplot(y=df_grouped[col], ax=axes[i], width=0.5, color='skyblue')
    axes[i].set_title(f'Mean {protein_name} Expression')
    axes[i].set_ylabel('Mean Expression Level')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
# Sanitize filename
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

for i, (col, protein_name) in enumerate(group2):
    sns.boxplot(y=df_grouped[col], ax=axes[i], width=0.5, color='skyblue')
    axes[i].set_title(f'Mean {protein_name} Expression')
    axes[i].set_ylabel('Mean Expression Level')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
# Sanitize filename
output_path_png = os.path.join(OUTPUT_DIR, 'merged_mean_boxplot_group2.png')
output_path_pdf = os.path.join(OUTPUT_DIR, 'merged_mean_boxplot_group2.pdf')
plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
plt.savefig(output_path_pdf, bbox_inches='tight')
plt.close()

# Perform EDA for each clinical variable after grouping by patient ID
print("Performing EDA for each clinical variable after grouping by patient ID...")
clinical_cols = [col for col in df.columns if col not in protein_cols and col != 'PID']
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
summary_stats = df[protein_cols].describe()
summary_stats.columns = [col_to_protein.get(col, col) for col in protein_cols]
summary_stats.to_csv(os.path.join(OUTPUT_DIR, 'protein_summary_statistics.csv'))

# Create correlation heatmap
print("Creating correlation heatmap...")
plt.figure(figsize=(15, 12))
correlation_matrix = df[protein_cols].corr()
correlation_matrix.columns = [col_to_protein.get(col, col) for col in protein_cols]
correlation_matrix.index = [col_to_protein.get(col, col) for col in protein_cols]
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Protein Expression Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'protein_correlation_heatmap.png'), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'protein_correlation_heatmap.pdf'))
plt.close()

# Perform EDA for clinical variables: box plots of numeric variables grouped by categorical variables
print("Performing EDA for clinical variables (numeric vs categorical)...")

# Define clinical variables (customize as needed)
categorical_vars = [
    'clinical_type', 'grade'
]
numeric_vars = [
    'age', 'OSmonth'
]

# Define order for clinical types
clinical_type_order = ['TripleNeg', 'HR-HER2+', 'HR+HER2-', 'HR+HER2+']

# Define darker shades for clinical types
my_pal = {"TripleNeg": "#4682B4",  # Steel Blue (Darker Blue)
          "HR-HER2+": "#DAA520",   # Goldenrod (Darker Yellow)
          "HR+HER2-": "#228B22",   # Forest Green (Darker Green)
          "HR+HER2+": "#B22222"}   # Firebrick (Darker Red)

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
            
            # Use custom order and colors for clinical_type
            if cat_var == 'clinical_type':
                sns.boxplot(x=plot_df[cat_var], y=plot_df[num_var], hue=plot_df[cat_var], palette=my_pal, order=clinical_type_order, legend=False)
            else:
                sns.boxplot(x=plot_df[cat_var], y=plot_df[num_var], color='skyblue')
                
            plt.title(f'{num_var} by {cat_var}')
            plt.xlabel(cat_var)
            plt.ylabel(num_var)
            plt.tight_layout()
            plot_path_png = os.path.join(OUTPUT_DIR, f'{sanitize_filename(num_var)}_by_{sanitize_filename(cat_var)}_boxplot.png')
            plot_path_pdf = os.path.join(OUTPUT_DIR, f'{sanitize_filename(num_var)}_by_{sanitize_filename(cat_var)}_boxplot.pdf')
            plt.savefig(plot_path_png, dpi=300)
            plt.savefig(plot_path_pdf)
            plt.close()
            # Summary statistics per category
            stats = plot_df.groupby(cat_var)[num_var].describe()
            stats.to_csv(os.path.join(OUTPUT_DIR, f'{sanitize_filename(num_var)}_by_{sanitize_filename(cat_var)}_summary.csv'))

# Add scatter plots for each marker vs OSmonth
print("Creating scatter plots for each marker vs OSmonth...")

# Calculate number of rows and columns for the grid
n_markers = len(col_to_protein)
n_cols = 6  # You can adjust this number to change the layout
n_rows = (n_markers + n_cols - 1) // n_cols

# Create a single figure for all scatter plots
plt.figure(figsize=(30, 5*n_rows))
for i, (col, protein_name) in enumerate(col_to_protein.items()):
    if col in df_grouped.columns and 'OSmonth' in df_grouped.columns:
        print(f"Creating scatter plot for {protein_name} vs OSmonth")
        
        # Calculate correlation coefficient
        correlation = df_grouped[col].corr(df_grouped['OSmonth'])
        
        # Create subplot
        plt.subplot(n_rows, n_cols, i+1)
        sns.regplot(x=df_grouped['OSmonth'], y=df_grouped[col], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'{protein_name} (r = {correlation:.3f})')
        plt.xlabel('OSmonth')
        plt.ylabel(f'Mean {protein_name} Expression')
        plt.tight_layout()

# Save the merged figure
plot_path_png = os.path.join(OUTPUT_DIR, 'merged_marker_vs_OSmonth_scatter.png')
plot_path_pdf = os.path.join(OUTPUT_DIR, 'merged_marker_vs_OSmonth_scatter.pdf')
plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
plt.savefig(plot_path_pdf, bbox_inches='tight')
plt.close()

print("Analysis complete! Results saved in:", OUTPUT_DIR) 