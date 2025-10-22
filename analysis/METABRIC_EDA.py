import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"

# Define get_gene_list function directly to avoid torch dependency
def get_gene_list(dataset_name="JacksonFischer"):
    if dataset_name == "METABRIC":
        return [
            'HH3_total', 'CK19', 'CK8_18', 'Twist', 'CD68', 'CK14', 'SMA', 'Vimentin',
            'c_Myc', 'HER2', 'CD3', 'HH3_ph', 'Erk1_2', 'Slug', 'ER', 'PR', 'p53', 'CD44',
            'EpCAM', 'CD45', 'GATA3', 'CD20', 'Beta_catenin', 'CAIX', 'E_cadherin', 'Ki67',
            'EGFR', 'pS6', 'Sox9', 'vWF_CD31', 'pmTOR', 'CK7', 'panCK', 'c_PARP_c_Casp3',
            'DNA1', 'DNA2', 'H3K27me3', 'CK5', 'Fibronectin'
        ]
    else:
        return []
DATA_PATH = os.path.join(BASE_DIR, "data/METABRIC/raw/merged_preprocessed_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/EDA/METABRIC")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define custom color palette for clinical subtypes
my_pal = {"TripleNeg": "#4682B4",  # Steel Blue (Darker Blue)
          "HR-HER2+": "#DAA520",   # Goldenrod (Darker Yellow)
          "HR+HER2-": "#228B22",   # Forest Green (Darker Green)
          "HR+HER2+": "#B22222"}   # Firebrick (Darker Red)

# Define the order of clinical subtypes
clinical_subtype_order = ["TripleNeg", "HR-HER2+", "HR+HER2-", "HR+HER2+"]

# Read the data
print("Reading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Define the classify_breast_cancer function
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

# Add the clinical_subtype column
df['clinical_subtype'] = df.apply(classify_breast_cancer, axis=1)

# Get marker names from custom_tools
marker_names = get_gene_list("METABRIC")

# Find all Intensity_MeanIntensity_FullStackc_* columns
intensity_cols = [col for col in df.columns if col.startswith('Intensity_MeanIntensity_FullStackc_')]

# Print for user verification
print("\nIntensity columns:", intensity_cols)
print("\nMarker names from get_gene_list:", marker_names)

# Try to map marker names to intensity columns by order
if len(marker_names) != len(intensity_cols):
    print(f"Warning: Number of marker names ({len(marker_names)}) does not match number of intensity columns ({len(intensity_cols)}). Please check the mapping!")
    # Print first few for manual inspection
    for i, (col, marker) in enumerate(zip(intensity_cols, marker_names)):
        print(f"{col} -> {marker}")
        if i > 10:
            break
else:
    print("Mapping Intensity columns to marker names by order.")

# Create mapping: intensity_col -> marker_name
col_to_marker = {}
for i, marker in enumerate(marker_names):
    if i < len(intensity_cols):
        col_to_marker[intensity_cols[i]] = marker
    else:
        print(f"No intensity column for marker {marker}")

# Use only the mapped columns for EDA
used_intensity_cols = list(col_to_marker.keys())
used_marker_names = [col_to_marker[col] for col in used_intensity_cols]

print("\nFinal mapping (first 10):")
for col, marker in list(col_to_marker.items())[:10]:
    print(f"{col} -> {marker}")

# Helper for plot filenames
def sanitize_filename(name):
    return re.sub(r'[^\w\-_\. ]', '_', name)

# Group by patient and calculate mean expression for each marker
print("Calculating mean expression per patient...")
df_grouped = df.groupby('PID')[used_intensity_cols + ['OSmonth']].mean().reset_index()

# Split markers into two groups for better visualization
half = len(used_intensity_cols) // 2
group1 = used_intensity_cols[:half]
group2 = used_intensity_cols[half:]

# Create a single figure with subplots for each marker in group 1
print("Creating merged box plot figure for group 1...")
n_markers1 = len(group1)
n_cols = 4
n_rows = (n_markers1 + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 25))
axes = axes.flatten()

for i, col in enumerate(group1):
    marker_name = col_to_marker[col]
    sns.boxplot(y=df_grouped[col], ax=axes[i], width=0.5, color='skyblue')
    axes[i].set_title(f'Mean {marker_name} Expression')
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
n_cols = 4
n_rows = (n_markers2 + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 25))
axes = axes.flatten()

for i, col in enumerate(group2):
    marker_name = col_to_marker[col]
    sns.boxplot(y=df_grouped[col], ax=axes[i], width=0.5, color='skyblue')
    axes[i].set_title(f'Mean {marker_name} Expression')
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
clinical_cols = [col for col in df.columns if col not in used_intensity_cols and col != 'PID']
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
summary_stats = df[used_intensity_cols]
summary_stats.columns = used_marker_names
summary_stats.describe().to_csv(os.path.join(OUTPUT_DIR, 'marker_summary_statistics.csv'))

# Create correlation heatmap
print("Creating correlation heatmap...")
plt.figure(figsize=(15, 12))
correlation_matrix = df[used_intensity_cols].corr()
correlation_matrix.columns = used_marker_names
correlation_matrix.index = used_marker_names
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Marker Expression Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'marker_correlation_heatmap.png'), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'marker_correlation_heatmap.pdf'))
plt.close()

# Perform EDA for clinical variables: box plots of numeric variables grouped by categorical variables
print("Performing EDA for clinical variables (numeric vs categorical)...")

# Define clinical variables (customize as needed)
categorical_vars = [
    'PAM50', 'grade', 'ER_Status', 'PR_Status', 'HER2_Status', 'clinical_subtype'
]
numeric_vars = [
    'age', 'OSmonth', 'tumor_size'
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
            
            # Use custom palette and order for clinical_subtype
            if cat_var == 'clinical_subtype':
                sns.boxplot(x=plot_df[cat_var], y=plot_df[num_var], palette=my_pal, order=clinical_subtype_order)
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
n_markers = len(used_intensity_cols)
n_cols = 6  # You can adjust this number to change the layout
n_rows = (n_markers + n_cols - 1) // n_cols

# Create a single figure for all scatter plots
plt.figure(figsize=(30, 5*n_rows))
for i, col in enumerate(used_intensity_cols):
    marker_name = col_to_marker[col]
    if col in df_grouped.columns and 'OSmonth' in df_grouped.columns:
        print(f"Creating scatter plot for {marker_name} vs OSmonth")
        
        # Calculate correlation coefficient
        correlation = df_grouped[col].corr(df_grouped['OSmonth'])
        
        # Create subplot
        plt.subplot(n_rows, n_cols, i+1)
        sns.regplot(x=df_grouped['OSmonth'], y=df_grouped[col], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'{marker_name} (r = {correlation:.3f})')
        plt.xlabel('OSmonth')
        plt.ylabel(f'Mean {marker_name} Expression')
        plt.tight_layout()

# Save the merged figure
plot_path_png = os.path.join(OUTPUT_DIR, 'merged_marker_vs_OSmonth_scatter.png')
plot_path_pdf = os.path.join(OUTPUT_DIR, 'merged_marker_vs_OSmonth_scatter.pdf')
plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
plt.savefig(plot_path_pdf, bbox_inches='tight')
plt.close()

# Image statistics analysis
print("Analyzing image statistics...")

# Total number of unique images (group by ImageNumber)
total_images = df['ImageNumber'].nunique()
print(f"Total number of unique images: {total_images}")

# Number of unique patients
unique_patients = df['PID'].nunique()
print(f"Total number of unique patients: {unique_patients}")

# Images per patient statistics (count unique ImageNumber per patient)
images_per_patient = df.groupby('PID')['ImageNumber'].nunique()
images_per_patient_stats = images_per_patient.describe()
print(f"\nImages per patient statistics:")
print(images_per_patient_stats)

# Save images per patient statistics
images_per_patient_stats.to_csv(os.path.join(OUTPUT_DIR, 'images_per_patient_statistics.csv'))

# Create histogram of images per patient
plt.figure(figsize=(10, 6))
plt.hist(images_per_patient, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Images per Patient')
plt.xlabel('Number of Images')
plt.ylabel('Number of Patients')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'images_per_patient_histogram.png'), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'images_per_patient_histogram.pdf'))
plt.close()

# Create box plot of images per patient
plt.figure(figsize=(8, 6))
sns.boxplot(y=images_per_patient, color='skyblue')
plt.title('Images per Patient Distribution')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'images_per_patient_boxplot.png'), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, 'images_per_patient_boxplot.pdf'))
plt.close()

# Images per patient by clinical subtype
if 'clinical_subtype' in df.columns:
    print("\nAnalyzing images per patient by clinical subtype...")
    
    # Calculate images per patient for each clinical subtype
    subtype_image_stats = []
    for subtype in clinical_subtype_order:
        if subtype in df['clinical_subtype'].values:
            subtype_df = df[df['clinical_subtype'] == subtype]
            subtype_images_per_patient = subtype_df.groupby('PID')['ImageNumber'].nunique()
            subtype_stats = subtype_images_per_patient.describe()
            subtype_stats.name = subtype
            subtype_image_stats.append(subtype_stats)
    
    # Combine statistics
    if subtype_image_stats:
        subtype_image_df = pd.concat(subtype_image_stats, axis=1)
        subtype_image_df.to_csv(os.path.join(OUTPUT_DIR, 'images_per_patient_by_clinical_subtype.csv'))
        print("Images per patient by clinical subtype:")
        print(subtype_image_df)
        
        # Create box plot of images per patient by clinical subtype
        plt.figure(figsize=(12, 8))
        plot_data = []
        plot_labels = []
        plot_colors = []
        
        for subtype in clinical_subtype_order:
            if subtype in df['clinical_subtype'].values:
                subtype_df = df[df['clinical_subtype'] == subtype]
                subtype_images = subtype_df.groupby('PID')['ImageNumber'].nunique()
                plot_data.append(subtype_images)
                plot_labels.append(subtype)
                plot_colors.append(my_pal[subtype])
        
        if plot_data:
            # Create box plot manually to avoid seaborn issues
            bp = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], plot_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.title('Images per Patient by Clinical Subtype')
            plt.xlabel('Clinical Subtype')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'images_per_patient_by_clinical_subtype_boxplot.png'), dpi=300)
            plt.savefig(os.path.join(OUTPUT_DIR, 'images_per_patient_by_clinical_subtype_boxplot.pdf'))
            plt.close()

# Read existing dataset statistics if it exists
existing_stats_path = os.path.join(OUTPUT_DIR, 'dataset_statistics.csv')
if os.path.exists(existing_stats_path):
    existing_df = pd.read_csv(existing_stats_path)
    print("Found existing dataset_statistics.csv, appending image statistics...")
else:
    existing_df = pd.DataFrame(columns=['Metric', 'Value'])

# Calculate additional statistics
print("Calculating additional dataset statistics...")

# Cells per image statistics
cells_per_image = df.groupby('ImageNumber').size()
cells_per_image_stats = cells_per_image.describe()

# Cells per sample (patient) statistics  
cells_per_sample = df.groupby('PID').size()
cells_per_sample_stats = cells_per_sample.describe()

# Summary statistics for the dataset
dataset_summary = {
    'Total number of patients': unique_patients,
    'Total number of images': total_images,
    'Average cells per image': cells_per_image.mean(),
    'Standard deviation of cells per image': cells_per_image.std(),
    'Minimum cells per image': cells_per_image.min(),
    'Maximum cells per image': cells_per_image.max(),
    'Minimum cells per sample': cells_per_sample.min(),
    'Maximum cells per sample': cells_per_sample.max(),
    'Minimum images per patient': images_per_patient.min(),
    'Maximum images per patient': images_per_patient.max(),
    'Mean Images per Patient': images_per_patient.mean(),
    'Median Images per Patient': images_per_patient.median(),
    'Std Images per Patient': images_per_patient.std()
}

# Create new statistics dataframe
new_stats_df = pd.DataFrame(list(dataset_summary.items()), columns=['Metric', 'Value'])

# Combine existing and new statistics
combined_df = pd.concat([existing_df, new_stats_df], ignore_index=True)

# Save combined dataset summary
combined_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_statistics.csv'), index=False)

print(f"\nDataset Summary:")
for metric, value in dataset_summary.items():
    print(f"{metric}: {value}")

print(f"\nCells per image statistics:")
print(cells_per_image_stats)

print(f"\nCells per sample (patient) statistics:")
print(cells_per_sample_stats)

print("Analysis complete! Results saved in:", OUTPUT_DIR) 