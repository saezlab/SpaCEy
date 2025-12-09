library(SingleCellExperiment)

setwd("./data/Lung")
sce <- readRDS("sce_all_annotated.rds")
# Set a chunk size (e.g., 1000 cells per chunk)
# Define the columns you want to extract from colData
columns_to_extract <- c("ImageNumber", "CellNumber", "Center_X", "Center_Y", "Area", 
                        "Compartment", "Area_Description", "BatchID", "CellID", 
                        "cell_category", "cell_type", "cell_subtype", "Patient_Nr", 
                        "Age", "Gender", "Typ", "Grade", "Size", "Vessel", "Pleura", 
                        "T.new", "N", "M.new", "Stage", "R", "Relapse",
                        "DFS", "OS", "Patient_ID")

# Extract only these columns from colData
metadata_subset <- colData(sce)[, columns_to_extract]

# Set chunk size (e.g., 1000 cells per chunk)
chunk_size <- 1000

# Get the number of cells
num_cells <- ncol(sce)

# Initialize the output file with the column names
write.table(metadata_subset[1:chunk_size, ], file = "sce_counts_metadata_subset.csv", 
            sep = ",", row.names = TRUE, col.names = TRUE)

# Loop through the counts and write in chunks
for (start_col in seq(1, num_cells, by = chunk_size)) {
  end_col <- min(start_col + chunk_size - 1, num_cells)
  
  # Extract the counts for the current chunk of cells
  chunk_counts <- t(assay(sce, "counts")[, start_col:end_col])  # transpose for (rows = cells)
  
  # Convert to data frame
  chunk_counts_df <- as.data.frame(chunk_counts)
  
  # Combine with the corresponding metadata for these cells
  chunk_metadata_df <- metadata_subset[start_col:end_col, ]
  final_chunk_df <- cbind(chunk_metadata_df, chunk_counts_df)
  
  # Append the chunk to the CSV
  write.table(final_chunk_df, file = "sce_counts_metadata_subset.csv", sep = ",", 
              append = TRUE, col.names = FALSE, row.names = TRUE)
}



# Load the RDS file if you haven't already
sce <- readRDS("sce_all_annotated.rds")

# Extract the counts matrix
counts_matrix <- assay(sce, "counts")  # rows = markers, columns = cells

# Transpose so rows = cells, columns = markers
counts_df <- as.data.frame(t(counts_matrix))

# Save to CSV
write.csv(counts_df, file = "sce_counts_matrix.csv", row.names = TRUE)