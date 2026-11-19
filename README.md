# Prediction of Clinical Features Using Graph Neural Networks

Discovery of functional motifs by association to clinical features using Graph Neural Networks. 

## Installation

We use conda. The environment file below has the versions we tested (Python 3.11, PyTorch 2.1.1 with CUDA 12.1,
PyTorch Geometric 2.4.0):

```bash
conda env create -f pygem_environment.yml
conda activate py_gem_3
```

Training needs a GPU; it stops with an error if PyTorch cannot see one. On NixOS, PyTorch only finds the
driver when you set:

```bash
export LD_LIBRARY_PATH=/run/opengl-driver/lib
```

## Data

Each dataset lives in `data/<dataset>/`.

### Main options

| Option | What it does | Default |
|---|---|---|
| `--dataset_name` | `JacksonFischer` or `METABRIC` | `JacksonFischer` |
| `--en` | experiment name, used for the output folders | `test_experiment` |
| `--folds_json` | the folds file (required, except with `--full_training`) | |
| `--full_training` | train one model on all patients instead of cross-validating, for example to explain it afterwards. The number of epochs is chosen on 20% of the patients, then the model is retrained on everyone. Saves `models/<en>/<session>_SD.mdl` and `<session>.json` | off |
| `--params_json` | hyperparameter file; its values replace the ones given on the command line | |
| `--cv_setting` | `nfold` or `heldout` (see above) | `nfold` |
| `--feature_normalise` | `raw`: use the protein abundances as they are; `log1p_z`: log-transform them and scale them with the training-fold mean and SD | `raw` |
| `--seed` | fold *k* is trained with seed `seed + k` | `42` |
| `--gpu_id` | which GPU to use | `0` |

### Hyperparameters

You can set these on the command line, but usually they come from the hyperparameter file.

| Hyperparameter | Meaning | Default |
|---|---|---|
| `model` | graph layer type: `GATV2`, `GATConv`, `PNAConv`, `TransformerConv` or `GINConv` | `GATV2` |
| `num_of_gcn_layers` | number of graph layers | 2 |
| `gcn_h` | size of each graph layer | 128 |
| `heads` | attention heads (GAT layers only) | 1 |
| `num_of_ff_layers` | number of fully connected layers after the graph layers | 2 |
| `fcl` | size of the fully connected layers | 128 |
| `dropout` | dropout rate | 0.2 |
| `lr` | learning rate | 0.01 |
| `weight_decay` | L2 penalty | 0.001 |
| `bs` | batch size (images) | 32 |
| `epoch` | maximum number of epochs | 5 |
| `factor`, `patience`, `min_lr` | the learning rate is multiplied by `factor` after `patience` epochs without improvement, down to `min_lr`. Training stops after 2 × `patience` epochs without improvement | 0.5, 5, 0.00002 |
| `aggregators`, `scalers` | aggregation settings for `PNAConv` | `sum mean`, `amplification identity` |
| `use_batch_norm` | batch normalisation after each graph layer (file only) | false |
| `pooling` | how cells are summarised into one image vector: `mean` or `mean_max` (file only) | `mean` |
| `augment` | randomly perturb the graphs during training (`--augment` on the command line) | false |
| `min_epochs` | smallest epoch that can be kept (`--min_epochs`) | 0 |



### Using another GPU

To train on a second GPU, make only that card visible and keep `--gpu_id 0`:

```bash
CUDA_VISIBLE_DEVICES=1 python train_test_controller.py ... --gpu_id 0
```

## Classification

Classification tasks (for example tumour grade) use a separate script with its own options:

```bash
python bin/train_test_controller_classification.py --help
```

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).
