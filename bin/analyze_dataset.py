import pandas as pd
import numpy as np
import os


def analyze_dataset_JascksonFischer():
    # Read the dataset
    df = pd.read_csv('./data/JacksonFischer/raw/merged_preprocessed_dataset.csv', low_memory=False)

    # Calculate summary statistics
    total_patients = df['PID'].nunique()
    total_images = df['ImageNumber'].nunique()
    avg_cells_per_image = df.groupby('ImageNumber').size().mean()
    std_cells_per_image = df.groupby('ImageNumber').size().std()
    min_cells_per_image = df.groupby('ImageNumber').size().min()
    max_cells_per_image = df.groupby('ImageNumber').size().max()

    # Calculate average cells per patient
    avg_cells_per_patient = df.groupby('PID').size().mean()
    std_cells_per_patient = df.groupby('PID').size().std()
    min_cells_per_patient = df.groupby('PID').size().min()
    max_cells_per_patient = df.groupby('PID').size().max()

    # Calculate average images per patient
    avg_images_per_patient = df.groupby('PID')['ImageNumber'].nunique().mean()
    std_images_per_patient = df.groupby('PID')['ImageNumber'].nunique().std()
    min_images_per_patient = df.groupby('PID')['ImageNumber'].nunique().min()
    max_images_per_patient = df.groupby('PID')['ImageNumber'].nunique().max()

    # Calculate clinical subtype statistics
    clinical_subtype_counts = df.groupby('clinical_type')['PID'].nunique().reset_index()
    clinical_subtype_counts.columns = ['Clinical Type', 'Number of Patients']

    # Create a dictionary with the general statistics
    general_stats_dict = {
        'Metric': [
            'Total number of patients',
            'Total number of images',
            'Average cells per image',
            'Standard deviation of cells per image',
            'Minimum cells per image',
            'Maximum cells per image',
            'Average cells per patient',
            'Standard deviation of cells per patient',
            'Minimum cells per patient',
            'Maximum cells per patient',
            'Average images per patient',
            'Standard deviation of images per patient',
            'Minimum images per patient',
            'Maximum images per patient'
        ],
        'Value': [
            total_patients,
            total_images,
            f"{avg_cells_per_image:.2f}",
            f"{std_cells_per_image:.2f}",
            min_cells_per_image,
            max_cells_per_image,
            f"{avg_cells_per_patient:.2f}",
            f"{std_cells_per_patient:.2f}",
            min_cells_per_patient,
            max_cells_per_patient,
            f"{avg_images_per_patient:.2f}",
            f"{std_images_per_patient:.2f}",
            min_images_per_patient,
            max_images_per_patient
        ]
    }

    # Create DataFrames
    general_stats_df = pd.DataFrame(general_stats_dict)

    # Create directory if it doesn't exist
    output_dir = './plots/analysis/EDA/JacksonFischer'
    os.makedirs(output_dir, exist_ok=True)

    # Save merged statistics to CSV
    output_file = os.path.join(output_dir, 'dataset_statistics.csv')

    # Create a list to store all rows
    all_rows = []

    # Add general statistics
    for _, row in general_stats_df.iterrows():
        all_rows.append({'Category': 'General Statistics', 'Metric': row['Metric'], 'Value': row['Value']})

    # Add clinical subtype statistics
    all_rows.append({'Category': 'Clinical Subtype Distribution', 'Metric': 'Header', 'Value': 'Header'})
    for _, row in clinical_subtype_counts.iterrows():
        all_rows.append({
            'Category': 'Clinical Subtype Distribution',
            'Metric': row['Clinical Type'],
            'Value': row['Number of Patients']
        })

    # Create final DataFrame and save
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(output_file, index=False)

    # Print summary statistics
    print("\nDataset Summary Statistics:")
    print("=" * 50)
    print(f"Total number of patients: {total_patients}")
    print(f"Total number of images: {total_images}")
    print("\nCells per Image Statistics:")
    print(f"Average cells per image: {avg_cells_per_image:.2f}")
    print(f"Standard deviation: {std_cells_per_image:.2f}")
    print(f"Minimum cells per image: {min_cells_per_image}")
    print(f"Maximum cells per image: {max_cells_per_image}")
    print("\nCells per Patient Statistics:")
    print(f"Average cells per patient: {avg_cells_per_patient:.2f}")
    print(f"Standard deviation: {std_cells_per_patient:.2f}")
    print(f"Minimum cells per patient: {min_cells_per_patient}")
    print(f"Maximum cells per patient: {max_cells_per_patient}")
    print("\nImages per Patient Statistics:")
    print(f"Average images per patient: {avg_images_per_patient:.2f}")
    print(f"Standard deviation: {std_images_per_patient:.2f}")
    print(f"Minimum images per patient: {min_images_per_patient}")
    print(f"Maximum images per patient: {max_images_per_patient}")

    print("\nClinical Subtype Distribution:")
    print("=" * 50)
    print(clinical_subtype_counts.to_string(index=False))

    print(f"\nAll statistics have been saved to: {output_file}")


def analyze_dataset_METABRIC():
    # Read the dataset
    df = pd.read_csv('./data/METABRIC/raw/merged_preprocessed_dataset.csv', low_memory=False)

    # Ensure clinical_subtype column exists (create if not present)
    if 'clinical_subtype' not in df.columns:
        def classify_breast_cancer(row):
            er = row.get('ER Status', None)
            pr = row.get('PR Status', None)
            her2 = row.get('HER2 Status', None)
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
                return np.nan
        df['clinical_subtype'] = df.apply(classify_breast_cancer, axis=1)

    # Calculate summary statistics
    total_patients = df['PID'].nunique()
    avg_cells_per_patient = df.groupby('PID').size().mean()
    std_cells_per_patient = df.groupby('PID').size().std()
    min_cells_per_patient = df.groupby('PID').size().min()
    max_cells_per_patient = df.groupby('PID').size().max()

    # Calculate clinical subtype statistics
    clinical_subtype_counts = df.groupby('clinical_subtype')['PID'].nunique().reset_index()
    clinical_subtype_counts.columns = ['Clinical Subtype', 'Number of Patients']

    # Create a dictionary with the general statistics
    general_stats_dict = {
        'Metric': [
            'Total number of patients',
            'Average cells per patient',
            'Standard deviation of cells per patient',
            'Minimum cells per patient',
            'Maximum cells per patient'
        ],
        'Value': [
            total_patients,
            f"{avg_cells_per_patient:.2f}",
            f"{std_cells_per_patient:.2f}",
            min_cells_per_patient,
            max_cells_per_patient
        ]
    }

    # Create DataFrames
    general_stats_df = pd.DataFrame(general_stats_dict)

    # Create directory if it doesn't exist
    output_dir = './plots/analysis/EDA/METABRIC'
    os.makedirs(output_dir, exist_ok=True)

    # Save merged statistics to CSV
    output_file = os.path.join(output_dir, 'dataset_statistics.csv')

    # Create a list to store all rows
    all_rows = []

    # Add general statistics
    for _, row in general_stats_df.iterrows():
        all_rows.append({'Category': 'General Statistics', 'Metric': row['Metric'], 'Value': row['Value']})

    # Add clinical subtype statistics
    all_rows.append({'Category': 'Clinical Subtype Distribution', 'Metric': 'Header', 'Value': 'Header'})
    for _, row in clinical_subtype_counts.iterrows():
        all_rows.append({
            'Category': 'Clinical Subtype Distribution',
            'Metric': row['Clinical Subtype'],
            'Value': row['Number of Patients']
        })

    # Create final DataFrame and save
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(output_file, index=False)

    # Print summary statistics
    print("\nMETABRIC Dataset Summary Statistics:")
    print("=" * 50)
    print(f"Total number of patients: {total_patients}")
    print("\nCells per Patient Statistics:")
    print(f"Average cells per patient: {avg_cells_per_patient:.2f}")
    print(f"Standard deviation: {std_cells_per_patient:.2f}")
    print(f"Minimum cells per patient: {min_cells_per_patient}")
    print(f"Maximum cells per patient: {max_cells_per_patient}")

    print("\nClinical Subtype Distribution:")
    print("=" * 50)
    print(clinical_subtype_counts.to_string(index=False))

    print(f"\nAll statistics have been saved to: {output_file}")


if __name__ == "__main__":
    analyze_dataset_METABRIC()


