# Hyperparameter Search Space for Supplementary Material

## Table 1: Hyperparameter Grid Search Configuration

| Hyperparameter | Values | Description |
|---|---|---|
| **Model Architecture** | | |
| Number of GCN Layers | 2 | Number of graph convolutional layers |
| Number of Feed-Forward Layers | 1, 2 | Number of fully connected layers |
| GCN Hidden Dimension | 16, 32, 64, 128 | Hidden dimension size for graph convolutional layers |
| Fully Connected Layer Dimension | 64, 128, 256 | Hidden dimension size for fully connected layers |
| Dropout Rate | 0.2, 0.3 | Dropout probability for regularization |
| **Training Parameters** | | |
| Learning Rate | 0.1, 0.01, 0.001, 0.0001 | Initial learning rate for optimization |
| Batch Size | 16, 32, 64 | Number of samples per training batch |
| Epochs | 200 | Maximum number of training epochs |
| Weight Decay | 0.1, 0.001, 0.0001, 1e-5 | L2 regularization coefficient |
| **Learning Rate Scheduler** | | |
| Factor | 0.5, 0.8, 0.2 | Factor by which learning rate is reduced |
| Patience | 5, 10 | Number of epochs with no improvement before reducing LR |
| Minimum Learning Rate | 0.00002, 0.0001 | Lower bound for learning rate |
| **Model-Specific Parameters** | | |
| GAT Heads | 1, 3, 5 | Number of attention heads (GAT model only) |
| PNA Aggregators | min, max, sum, mean, sum max | Aggregation functions (PNA model only) |
| PNA Scalers | identity, amplification | Scaling functions (PNA model only) |

---

## Table 2: Total Hyperparameter Combinations

| Model | Base Combinations | Model-Specific Combinations | Total Combinations |
|---|---|---|---|
| GCN | 4 × 3 × 2 × 2 × 4 × 3 × 4 × 3 × 3 × 2 × 2 | - | 82,944 |
| GATConv | 4 × 3 × 2 × 2 × 4 × 3 × 4 × 3 × 3 × 2 × 2 | × 3 (heads) | 248,832 |
| TransformerConv | 4 × 3 × 2 × 2 × 4 × 3 × 4 × 3 × 3 × 2 × 2 | - | 82,944 |
| PNAConv | 4 × 3 × 2 × 2 × 4 × 3 × 4 × 3 × 3 × 2 × 2 | × 5 (agg) × 2 (scaler) | 829,440 |

**Note:** Base combinations calculated from: lr (4) × bs (3) × dropout (2) × num_ff_layers (2) × gcn_h (4) × fcl (3) × weight_decay (4) × factor (3) × patience (2) × min_lr (2) × num_gcn_layers (1) × epoch (1)

---

## LaTeX Table Format (for direct copy-paste into manuscript)

```latex
\begin{table}[htbp]
\centering
\caption{Hyperparameter grid search space used for model optimization.}
\label{tab:hyperparameters}
\begin{tabular}{llp{6cm}}
\hline
\textbf{Hyperparameter} & \textbf{Values} & \textbf{Description} \\
\hline
\multicolumn{3}{l}{\textit{Model Architecture}} \\
Number of GCN Layers & 2 & Number of graph convolutional layers \\
Number of FF Layers & 1, 2 & Number of fully connected layers \\
GCN Hidden Dimension & 16, 32, 64, 128 & Hidden dimension for GCN layers \\
FC Layer Dimension & 64, 128, 256 & Hidden dimension for FC layers \\
Dropout Rate & 0.2, 0.3 & Dropout probability \\
\hline
\multicolumn{3}{l}{\textit{Training Parameters}} \\
Learning Rate & 0.1, 0.01, 0.001, 0.0001 & Initial learning rate \\
Batch Size & 16, 32, 64 & Samples per batch \\
Epochs & 200 & Maximum training epochs \\
Weight Decay & 0.1, 0.001, 0.0001, $1 \times 10^{-5}$ & L2 regularization \\
\hline
\multicolumn{3}{l}{\textit{Learning Rate Scheduler}} \\
Factor & 0.5, 0.8, 0.2 & LR reduction factor \\
Patience & 5, 10 & Epochs before LR reduction \\
Minimum LR & 0.00002, 0.0001 & Lower bound for LR \\
\hline
\multicolumn{3}{l}{\textit{Model-Specific Parameters}} \\
GAT Heads & 1, 3, 5 & Attention heads (GAT only) \\
PNA Aggregators & min, max, sum, mean, sum max & Aggregators (PNA only) \\
PNA Scalers & identity, amplification & Scalers (PNA only) \\
\hline
\end{tabular}
\end{table}
```

---

## CSV Format (for Excel/Google Sheets)

```csv
Category,Hyperparameter,Values,Description
Model Architecture,Number of GCN Layers,2,Number of graph convolutional layers
Model Architecture,Number of Feed-Forward Layers,"1, 2",Number of fully connected layers
Model Architecture,GCN Hidden Dimension,"16, 32, 64, 128",Hidden dimension size for graph convolutional layers
Model Architecture,Fully Connected Layer Dimension,"64, 128, 256",Hidden dimension size for fully connected layers
Model Architecture,Dropout Rate,"0.2, 0.3",Dropout probability for regularization
Training Parameters,Learning Rate,"0.1, 0.01, 0.001, 0.0001",Initial learning rate for optimization
Training Parameters,Batch Size,"16, 32, 64",Number of samples per training batch
Training Parameters,Epochs,200,Maximum number of training epochs
Training Parameters,Weight Decay,"0.1, 0.001, 0.0001, 1e-5",L2 regularization coefficient
Learning Rate Scheduler,Factor,"0.5, 0.8, 0.2",Factor by which learning rate is reduced
Learning Rate Scheduler,Patience,"5, 10",Number of epochs with no improvement before reducing LR
Learning Rate Scheduler,Minimum Learning Rate,"0.00002, 0.0001",Lower bound for learning rate
Model-Specific,GAT Heads,"1, 3, 5",Number of attention heads (GAT model only)
Model-Specific,PNA Aggregators,"min, max, sum, mean, sum max",Aggregation functions (PNA model only)
Model-Specific,PNA Scalers,"identity, amplification",Scaling functions (PNA model only)
```


