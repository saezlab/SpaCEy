# Prediction of Clinical Features Using Graph Neural Networks

Discovery of functional motifs by association to clinical features using Graph Neural Networks. 

## Running 

### Training PNA For Regression

```bash
python train_test_controller.py --aggregators 'max' --bs 16 --dropout 0.0 --en my_experiment --epoch 200 --factor 0.8 --fcl 256 --gcn_h 64 --lr 0.001 --min_lr 0.0001 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --scalers 'identity' --weight_decay 1e-05
```
### Training GAT For Regression
```bash
python train_test_controller.py --aggregators None --bs 16 --dropout 0.0 --en my_experiment --epoch 200 --factor 0.2 --fcl 128 --gcn_h 64 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --scalers None --weight_decay 0
```


### GNNExplainer For PNA Regressor
```bash
python gnnexplainer.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 10
```
### LIME Explainer for PNA Regressor
```bash
python lime.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 10
```

### SHAP Explainer for PNA Regressor
```bash
python shap.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 10
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

For a full list, run:
```bash
python bin/train_test_controller_classification.py --help
```



