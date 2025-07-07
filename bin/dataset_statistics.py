import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

def get_dataset_statistics():
    # Path to the raw data directory
    raw_data_path = "../data/JacksonFischer/raw"
    
    # Initialize counters and storage
    patient_ids = set()
    image_ids = set()
    cells_per_image = []
    cells_per_patient = defaultdict(int)
    
    # Process each feature file
    for filename in os.listdir(raw_data_path):
        if filename.endswith("_features.pickle"):
            # Extract patient ID and image ID
            parts = filename.split("_")
            if len(parts) >= 2:
                img_id = parts[0]
                pid = parts[1].split("-")[0]  # Remove location suffix if present
                
                # Add to sets
                patient_ids.add(pid)
                image_ids.add(img_id)
                
                # Count cells in this image
                with open(os.path.join(raw_data_path, filename), 'rb') as handle:
                    features = pickle.load(handle)
                    num_cells = len(features)
                    cells_per_image.append(num_cells)
                    cells_per_patient[pid] += num_cells
    
    # Calculate statistics
    stats = {
        "Number of Patients": len(patient_ids),
        "Number of Images": len(image_ids),
        "Total Number of Cells": sum(cells_per_image),
        "Average Cells per Image": np.mean(cells_per_image),
        "Median Cells per Image": np.median(cells_per_image),
        "Min Cells per Image": min(cells_per_image),
        "Max Cells per Image": max(cells_per_image),
        "Average Cells per Patient": np.mean(list(cells_per_patient.values())),
        "Median Cells per Patient": np.median(list(cells_per_patient.values())),
        "Min Cells per Patient": min(cells_per_patient.values()),
        "Max Cells per Patient": max(cells_per_patient.values())
    }
    
    # Create DataFrame
    df_stats = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    
    # Save to CSV
    output_path = "../plots/dataset_statistics.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_stats.to_csv(output_path, index=False)
    
    return df_stats

if __name__ == "__main__":
    stats = get_dataset_statistics()
    print("\nDataset Statistics:")
    print(stats.to_string(index=False)) 