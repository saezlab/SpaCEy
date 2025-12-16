# Prediction of Clinical Features Using Graph Neural Networks

Discovery of functional motifs by association to clinical features using Graph Neural Networks. 

## Installation

### Using Conda (Recommended)

The project includes conda environment files. To set up the environment:

```bash
# Create environment from the provided yml file
conda env create -f pygem_environment.yml

# Activate the environment
conda activate py_gem_3
```

Alternatively, you can use the GPU-enabled environment:

```bash
conda env create -f environment.yml
conda activate py_gem_gpu
```

### Using Pip

Install the required packages manually:

```bash
pip install torch torch-geometric pytorch-lightning numpy pandas scikit-learn matplotlib seaborn scanpy networkx
```

### Key Dependencies

- Python 3.9+ or 3.11+
- PyTorch (>= 2.1.0)
- PyTorch Geometric (>= 2.4.0)
- PyTorch Lightning (>= 2.1.0)
- NumPy, Pandas, Scikit-learn
- Scanpy (for single-cell analysis)
- NetworkX (for graph operations)

## Training and Testing

The `bin/train_test_controller.py` script handles training and testing of GNN models for regression tasks. The script:

- Sets up experiment directories (results, plots, models)
- Configures loss functions (MSE, Huber, CoxPHLoss, NegativeLogLikelihood)
- Handles train/validation/test splits or cross-validation
- Uses PyTorch Lightning for training
- Saves models, results, and plots automatically

### Basic Usage

```bash
cd bin
python train_test_controller.py [arguments]
```

### Training PNA Model For Regression

```bash
python bin/train_test_controller.py \
    --aggregators 'max' \
    --bs 16 \
    --dropout 0.0 \
    --en my_experiment \
    --epoch 200 \
    --factor 0.8 \
    --fcl 256 \
    --gcn_h 64 \
    --lr 0.001 \
    --min_lr 0.0001 \
    --model PNAConv \
    --num_of_ff_layers 1 \
    --num_of_gcn_layers 2 \
    --patience 5 \
    --scalers 'identity' \
    --weight_decay 1e-05 \
    --loss MSE \
    --label OSMonth
```

### Training GAT Model For Regression

```bash
python bin/train_test_controller.py \
    --aggregators None \
    --bs 16 \
    --dropout 0.0 \
    --en my_experiment \
    --epoch 200 \
    --factor 0.2 \
    --fcl 128 \
    --gcn_h 64 \
    --lr 0.001 \
    --min_lr 2e-05 \
    --model GATConv \
    --num_of_ff_layers 1 \
    --num_of_gcn_layers 3 \
    --patience 20 \
    --scalers None \
    --weight_decay 0 \
    --loss MSE \
    --label OSMonth
```

### Key Features

- **Loss Functions**: Supports MSE, Huber, CoxPHLoss, and NegativeLogLikelihood
- **Models**: PNAConv, GATConv, GATv2Conv, and other PyTorch Geometric models
- **Cross-validation**: Use `--fold` flag for k-fold cross-validation
- **Experiment Tracking**: Results are saved with unique session IDs
- **Automatic Directory Creation**: Creates `results/`, `plots/`, and `models/` directories


### Model Explainability

The framework supports various explainability methods for understanding model predictions:

#### GNNExplainer For PNA Regressor
```bash
python bin/gnnexplainer.py \
    --aggregators 'max' \
    --bs 16 \
    --dropout 0.0 \
    --fcl 256 \
    --gcn_h 64 \
    --model PNAConv \
    --num_of_ff_layers 1 \
    --num_of_gcn_layers 2 \
    --scalers 'identity' \
    --idx 10
```

#### LIME Explainer for PNA Regressor
```bash
python bin/lime.py \
    --aggregators 'max' \
    --bs 16 \
    --dropout 0.0 \
    --fcl 256 \
    --gcn_h 64 \
    --model PNAConv \
    --num_of_ff_layers 1 \
    --num_of_gcn_layers 2 \
    --scalers 'identity' \
    --idx 10
```

#### SHAP Explainer for PNA Regressor
```bash
python bin/shap.py \
    --aggregators 'max' \
    --bs 16 \
    --dropout 0.0 \
    --fcl 256 \
    --gcn_h 64 \
    --model PNAConv \
    --num_of_ff_layers 1 \
    --num_of_gcn_layers 2 \
    --scalers 'identity' \
    --idx 10
```

## Explainable cells and cell interactions


| Original Graph                                                                                              | SubGraph                                                                                               |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Original Graph](https://github.com/saezlab/GNNClinicalOutcomePrediction/blob/main/plots/original_graphs/original_graph_28_50_0.001_regression_individual_feature.png) | ![QualitativeResults](https://github.com/saezlab/GNNClinicalOutcomePrediction/blob/main/plots/subgraphs/subgraph_28_50_0.001_regression_individual_feature.png) |

### First Results of Hyperparameter Tuning

![Explainer Results](https://github.com/saezlab/GNNClinicalOutcomePrediction/blob/main/plots/subgraphs/futon_explainer.gif)


## Classification Tasks (Binary & Multi-Class)

You can use this framework for both binary and multi-class classification (e.g., tumor grade prediction).

### Example: Tumor Grade Multi-Class Classification

```bash
python bin/train_test_controller_classification.py \
    --label tumor_grade \
    --model GATv2Conv \
    --bs 16 \
    --epoch 100 \
    --lr 0.001 \
    --en TumorGradeExp
```

- The script will automatically handle class weighting and AUC reporting for multi-class tasks.
- Results and per-fold AUCs will be saved and plotted automatically.

### Example: Binary Classification (Relapse)

```bash
python bin/train_test_controller_classification.py \
    --label Relapse \
    --model GATv2Conv \
    --bs 16 \
    --epoch 100 \
    --lr 0.001 \
    --en RelapseExp
```

## Hyperparameter Optimization with Optuna

You can run distributed hyperparameter optimization using Optuna. For tumor grade classification:

```bash
python bin/optuna_hpo_tumor_grade.py --n_trials 50 --n_jobs 1
```

- For SLURM clusters, launch multiple jobs with the same study name and storage backend (see Optuna docs for details).

## Main Hyperparameters

### Regression (train_test_controller.py)

| Argument         | Description                                      | Example Value      |
|------------------|--------------------------------------------------|--------------------|
| `--model`        | GNN model type (`PNAConv`, `GATConv`, `GATv2Conv`, etc.) | `PNAConv`        |
| `--label`        | Target label for regression (`OSMonth`, etc.)   | `OSMonth`          |
| `--loss`         | Loss function (`MSE`, `Huber`, `CoxPHLoss`, `NegativeLogLikelihood`) | `MSE` |
| `--bs`           | Batch size                                       | `16`               |
| `--epoch`        | Number of training epochs                        | `200`              |
| `--lr`           | Learning rate                                    | `0.001`            |
| `--min_lr`       | Minimum learning rate                            | `0.0001`           |
| `--factor`       | Learning rate reduction factor                   | `0.8`              |
| `--patience`     | Patience for learning rate scheduling            | `5`                |
| `--dropout`      | Dropout rate                                     | `0.0`              |
| `--fcl`          | Fully connected layer size                        | `256`              |
| `--gcn_h`        | GNN hidden layer size                            | `64`               |
| `--num_of_gcn_layers` | Number of GNN layers                         | `2`                |
| `--num_of_ff_layers`  | Number of fully connected layers              | `1`                |
| `--weight_decay` | L2 regularization weight                         | `1e-5`             |
| `--aggregators`  | Aggregators for PNAConv (`sum`, `mean`, `max`, `min`, `var`, `std`) | `'max'` |
| `--scalers`      | Scalers for PNAConv (`identity`, `amplification`, `attenuation`, etc.) | `'identity'` |
| `--heads`        | Number of attention heads (for GAT models)       | `1`                |
| `--dataset_name` | Dataset name (`JacksonFischer`, `METABRIC`, `Lung`) | `JacksonFischer` |
| `--fold`         | Use k-fold cross-validation (`--fold` or `--no-fold`) | `--fold` |
| `--en`           | Experiment name                                  | `my_experiment`    |
| `--gpu_id`       | GPU ID to use (0, 1, etc.)                       | `0`                |

### Classification (train_test_controller_classification.py)

| Argument         | Description                                      | Example Value      |
|------------------|--------------------------------------------------|--------------------|
| `--label`        | Target label (`tumor_grade`, `Relapse`, etc.)    | `tumor_grade`      |
| `--model`        | GNN model type (`GATv2Conv`, `PNAConv`, etc.)    | `GATv2Conv`        |
| `--bs`           | Batch size                                       | `16`               |
| `--epoch`        | Number of training epochs                        | `100`              |
| `--lr`           | Learning rate                                    | `0.001`            |
| `--dropout`      | Dropout rate                                     | `0.2`              |
| `--fcl`          | Fully connected layer size                        | `128`              |
| `--gcn_h`        | GNN hidden layer size                            | `64`               |
| `--num_of_gcn_layers` | Number of GNN layers                         | `2`                |
| `--num_of_ff_layers`  | Number of fully connected layers              | `1`                |
| `--weight_decay` | L2 regularization weight                         | `1e-5`             |
| `--en`           | Experiment name                                  | `TumorGradeExp`    |

For a full list of arguments, run:
```bash
# For regression
python bin/train_test_controller.py --help

# For classification
python bin/train_test_controller_classification.py --help
```



