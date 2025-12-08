import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict

# Set up paths
BASE_DIR = "/home/rifaioglu/projects/GNNClinicalOutcomePrediction"
DATA_PATH = os.path.join(BASE_DIR, "data/Lung/raw/merged_preprocessed_dataset.csv")
CLINICAL_DATA_PATH = os.path.join(BASE_DIR, "data/Lung/LUAD Clinical Data.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/Lung/raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots/analysis/EDA/Lung")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading merged dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Reading clinical data...")
clinical_df = pd.read_csv(CLINICAL_DATA_PATH)

# Filter dataset to only include patients in clinical data
print("\nFiltering dataset to only include patients in clinical data...")
clinical_patient_ids = set(clinical_df['Key'].unique())
print(f"Patients in clinical data: {len(clinical_patient_ids)}")
print(f"Patients in merged dataset before filtering: {df['sample_id'].nunique()}")

# Filter the merged dataset
df_filtered = df[df['sample_id'].isin(clinical_patient_ids)].copy()
print(f"Patients in merged dataset after filtering: {df_filtered['sample_id'].nunique()}")
print(f"Cells before filtering: {len(df):,}")
print(f"Cells after filtering: {len(df_filtered):,}")

# Use filtered dataset for all subsequent analysis
df = df_filtered

print("Analyzing raw data files to count images per patient...")
# Count images from pickle files
# Format: {image_id}_{patient_id}_features.pickle
# Example: LUAD_D001_LUAD_D001_features.pickle
# In most cases, image_id == patient_id, meaning 1 image per patient
# But we need to check if any patient has multiple images

patient_image_count = defaultdict(int)
image_ids = set()
patient_ids_from_files = set()

for filename in os.listdir(RAW_DATA_PATH):
    if filename.endswith("_features.pickle"):
        base_name = filename.replace("_features.pickle", "")
        # The pattern appears to be: LUAD_D001_LUAD_D001
        # Split and extract patient ID (the last LUAD_XXX part)
        parts = base_name.split("_")
        
        # Extract patient ID - it's the part that matches the pattern LUAD_XXX
        # Typically the last occurrence
        patient_id = None
        image_id = None
        
        # Look for LUAD pattern
        luad_indices = [i for i, part in enumerate(parts) if part == "LUAD"]
        
        if len(luad_indices) >= 2:
            # Pattern like: LUAD_D001_LUAD_D001
            # First LUAD_XXX is image, second is patient
            if luad_indices[0] + 1 < len(parts):
                image_id = f"{parts[luad_indices[0]]}_{parts[luad_indices[0] + 1]}"
            if luad_indices[1] + 1 < len(parts):
                patient_id = f"{parts[luad_indices[1]]}_{parts[luad_indices[1] + 1]}"
        elif len(luad_indices) == 1:
            # Only one LUAD pattern, use it for both
            if luad_indices[0] + 1 < len(parts):
                patient_id = f"{parts[luad_indices[0]]}_{parts[luad_indices[0] + 1]}"
                image_id = patient_id
        else:
            # Fallback: use the base name
            patient_id = base_name
            image_id = base_name
        
        # Only count images for patients in clinical data
        if patient_id and patient_id in clinical_patient_ids:
            image_ids.add(image_id if image_id else patient_id)
            patient_ids_from_files.add(patient_id)
            patient_image_count[patient_id] += 1

print(f"Found {len(image_ids)} unique images")
print(f"Found {len(patient_ids_from_files)} unique patients from files")

# Get patient statistics from the merged dataset
print("\nCalculating statistics from merged dataset...")
unique_patients = df['sample_id'].nunique()
total_cells = len(df)

# Count cells per patient
cells_per_patient = df.groupby('sample_id').size()
cells_per_patient_stats = cells_per_patient.describe()

# Count images per patient from file analysis
if patient_image_count:
    images_per_patient = pd.Series(patient_image_count)
    images_per_patient_stats = images_per_patient.describe()
else:
    # If we can't get image counts from files, assume 1 image per patient
    images_per_patient = pd.Series([1] * unique_patients)
    images_per_patient_stats = images_per_patient.describe()

# Create comprehensive statistics table
print("\nCreating statistics table...")

# Basic dataset statistics
dataset_stats = {
    'Metric': [
        'Total number of patients',
        'Total number of images',
        'Total number of cells',
        'Average cells per patient',
        'Median cells per patient',
        'Minimum cells per patient',
        'Maximum cells per patient',
        'Standard deviation of cells per patient',
        'Average images per patient',
        'Median images per patient',
        'Minimum images per patient',
        'Maximum images per patient',
        'Standard deviation of images per patient',
        'Average cells per image',
        'Median cells per image',
        'Minimum cells per image',
        'Maximum cells per image'
    ],
    'Value': [
        unique_patients,
        len(image_ids) if image_ids else unique_patients,
        total_cells,
        cells_per_patient.mean(),
        cells_per_patient.median(),
        cells_per_patient.min(),
        cells_per_patient.max(),
        cells_per_patient.std(),
        images_per_patient.mean(),
        images_per_patient.median(),
        images_per_patient.min(),
        images_per_patient.max(),
        images_per_patient.std(),
        total_cells / len(image_ids) if image_ids else cells_per_patient.mean(),
        np.nan,  # Will calculate separately
        np.nan,  # Will calculate separately
        np.nan   # Will calculate separately
    ]
}

# Calculate cells per image if we have image information
if image_ids and patient_image_count:
    # Estimate cells per image by dividing cells per patient by images per patient
    cells_per_image_list = []
    for patient_id in df['sample_id'].unique():
        patient_cells = len(df[df['sample_id'] == patient_id])
        patient_images = patient_image_count.get(patient_id, 1)
        cells_per_image_list.extend([patient_cells / patient_images] * patient_images)
    
    if cells_per_image_list:
        cells_per_image_series = pd.Series(cells_per_image_list)
        dataset_stats['Value'][13] = cells_per_image_series.mean()  # Average cells per image
        dataset_stats['Value'][14] = cells_per_image_series.median()  # Median cells per image
        dataset_stats['Value'][15] = cells_per_image_series.min()  # Minimum cells per image
        dataset_stats['Value'][16] = cells_per_image_series.max()  # Maximum cells per image

# Create DataFrame
stats_df = pd.DataFrame(dataset_stats)

# Save to CSV (filtered version)
output_path = os.path.join(OUTPUT_DIR, 'dataset_statistics_filtered.csv')
stats_df.to_csv(output_path, index=False)
print(f"\nDataset statistics (filtered to clinical data patients) saved to: {output_path}")

# Print statistics
print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
for _, row in stats_df.iterrows():
    value = row['Value']
    if pd.notna(value):
        if isinstance(value, float):
            print(f"{row['Metric']}: {value:.2f}")
        else:
            print(f"{row['Metric']}: {value}")
    else:
        print(f"{row['Metric']}: N/A")

# Create additional detailed tables
print("\nCreating detailed patient statistics...")

# Patient-level statistics
patient_stats = []
for patient_id in sorted(df['sample_id'].unique()):
    patient_data = df[df['sample_id'] == patient_id]
    num_images = patient_image_count.get(patient_id, 1)
    num_cells = len(patient_data)
    
    patient_stats.append({
        'Patient ID': patient_id,
        'Number of Images': num_images,
        'Number of Cells': num_cells,
        'Cells per Image': num_cells / num_images if num_images > 0 else num_cells
    })

patient_stats_df = pd.DataFrame(patient_stats)
patient_stats_path = os.path.join(OUTPUT_DIR, 'patient_level_statistics_filtered.csv')
patient_stats_df.to_csv(patient_stats_path, index=False)
print(f"Patient-level statistics (filtered) saved to: {patient_stats_path}")

# Clinical data summary
print("\nCreating clinical data summary...")
clinical_summary = {
    'Metric': [
        'Total patients in clinical data',
        'Patients with survival data',
        'Patients with death information',
        'Patients with progression information',
        'Patients with stage information',
        'Patients with histological pattern information'
    ],
    'Value': [
        len(clinical_df),
        clinical_df['Survival or loss to follow-up (years)'].notna().sum(),
        clinical_df['Death (No: 0, Yes: 1)'].notna().sum(),
        clinical_df['Progression'].notna().sum(),
        clinical_df['Stage (I-II: 0, III-IV:1)'].notna().sum(),
        clinical_df['Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)'].notna().sum()
    ]
}

clinical_summary_df = pd.DataFrame(clinical_summary)
clinical_summary_path = os.path.join(OUTPUT_DIR, 'clinical_data_summary.csv')
clinical_summary_df.to_csv(clinical_summary_path, index=False)
print(f"Clinical data summary saved to: {clinical_summary_path}")

# Create filtered summary table
print("\nCreating filtered summary table...")
filtered_summary = {
    'Statistic': [
        'Total number of patients (filtered)',
        'Total number of images (filtered)',
        'Minimum images per patient',
        'Maximum images per patient',
        'Average images per patient',
        'Median images per patient',
        'Standard deviation of images per patient',
        'Total number of cells (filtered)',
        'Average cells per patient',
        'Median cells per patient',
        'Minimum cells per patient',
        'Maximum cells per patient',
        'Standard deviation of cells per patient',
        'Average cells per image',
        'Median cells per image',
        'Minimum cells per image',
        'Maximum cells per image',
        'Total patients in clinical data',
        'Patients with survival data',
        'Patients with death information',
        'Patients with progression information',
        'Patients with stage information',
        'Patients with histological pattern information'
    ],
    'Value': [
        unique_patients,
        len(image_ids) if image_ids else unique_patients,
        images_per_patient.min(),
        images_per_patient.max(),
        images_per_patient.mean(),
        images_per_patient.median(),
        images_per_patient.std(),
        total_cells,
        cells_per_patient.mean(),
        cells_per_patient.median(),
        cells_per_patient.min(),
        cells_per_patient.max(),
        cells_per_patient.std(),
        dataset_stats['Value'][13] if len(dataset_stats['Value']) > 13 else np.nan,
        dataset_stats['Value'][14] if len(dataset_stats['Value']) > 14 else np.nan,
        dataset_stats['Value'][15] if len(dataset_stats['Value']) > 15 else np.nan,
        dataset_stats['Value'][16] if len(dataset_stats['Value']) > 16 else np.nan,
        len(clinical_df),
        clinical_df['Survival or loss to follow-up (years)'].notna().sum(),
        clinical_df['Death (No: 0, Yes: 1)'].notna().sum(),
        clinical_df['Progression'].notna().sum(),
        clinical_df['Stage (I-II: 0, III-IV:1)'].notna().sum(),
        clinical_df['Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)'].notna().sum()
    ]
}

filtered_summary_df = pd.DataFrame(filtered_summary)
filtered_summary_path = os.path.join(OUTPUT_DIR, 'dataset_summary_table_filtered.csv')
filtered_summary_df.to_csv(filtered_summary_path, index=False)
print(f"Filtered summary table saved to: {filtered_summary_path}")

print("\n" + "="*60)
print("CLINICAL DATA SUMMARY")
print("="*60)
for _, row in clinical_summary_df.iterrows():
    print(f"{row['Metric']}: {row['Value']}")

print("\nAnalysis complete!")

