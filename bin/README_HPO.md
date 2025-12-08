# Progression Classification Hyperparameter Optimization

This directory contains a comprehensive hyperparameter optimization system for the Progression classification task using Optuna.

## Files Overview

- `optuna_hpo_progression_full.py` - Main HPO implementation
- `run_progression_hpo.py` - Easy launcher script
- `monitor_hpo.py` - Progress monitoring tool
- `hpo_config.py` - Configuration definitions
- `README_HPO.md` - This documentation

## Quick Start

### 1. Quick Test (3 trials)
```bash
python run_progression_hpo.py --config quick
```

### 2. Standard Optimization (50 trials)
```bash
python run_progression_hpo.py --config standard
```

### 3. Comprehensive Optimization (100 trials)
```bash
python run_progression_hpo.py --config comprehensive
```

### 4. Production Optimization (200 trials)
```bash
python run_progression_hpo.py --config production
```

## Configuration Options

| Config | Trials | Epochs | Models | Use Case |
|--------|--------|--------|--------|----------|
| quick | 3 | 50 | GATV2 | Testing |
| standard | 50 | 200 | 4 models | Development |
| comprehensive | 100 | 300 | 5 models | Research |
| production | 200 | 500 | 7 models | Final optimization |

## Hyperparameter Search Space

### Model Architecture
- **Models**: GCN, GAT, GATV2, SAGE, PNAConv, MMAConv, GMNConv
- **Hidden sizes**: 16, 32, 64, 128, 256, 512
- **GCN layers**: 1-6
- **FF layers**: 1-5
- **Attention heads**: 1, 2, 4, 8

### Training Configuration
- **Learning rate**: 1e-6 to 1e-2 (log scale)
- **Weight decay**: 1e-6 to 1e-2 (log scale)
- **Batch size**: 8, 16, 32, 64, 128
- **Dropout**: 0.0 to 0.7
- **Patience**: 5-20 epochs

### Learning Rate Scheduling
- **Min learning rate**: 1e-6 to 1e-3 (log scale)
- **Reduction factor**: 0.1 to 0.8

## Monitoring Progress

### Real-time Monitoring
```bash
python monitor_hpo.py --watch
```

### Check Progress Once
```bash
python monitor_hpo.py
```

### Save Progress to CSV
```bash
python monitor_hpo.py --save_csv progress.csv
```

## Output Files

The optimization creates several output files in the `results/` directory:

1. **Log file**: `optuna_progression_full_YYYY-MM-DD_HH-MM_XXXXX.log`
   - Detailed training logs
   - Trial parameters and results
   - Best score tracking

2. **Study file**: `optuna_study_progression_hpo_YYYYMMDD_HHMMSS.json`
   - Complete Optuna study data
   - All trial results
   - Best parameters

3. **CSV file**: `trial_results_progression_hpo_YYYYMMDD_HHMMSS.csv`
   - Trial results in tabular format
   - Easy to analyze with pandas

## Custom Optimization

### Custom Number of Trials
```bash
python run_progression_hpo.py --custom --trials 25 --epochs 150
```

### Custom Study Name
```bash
python run_progression_hpo.py --config standard --study_name my_experiment
```

## Advanced Usage

### Direct Script Usage
```bash
python optuna_hpo_progression_full.py --n_trials 50 --study_name my_study
```

### Monitor Specific Log File
```bash
python monitor_hpo.py --log_file results/optuna_progression_full_2025-10-22_14-30_ABC123.log
```

## Results Analysis

### Load Results in Python
```python
import json
import pandas as pd

# Load study results
with open('results/optuna_study_progression_hpo_20251022_143000.json', 'r') as f:
    study_data = json.load(f)

# Load trial results
df = pd.read_csv('results/trial_results_progression_hpo_20251022_143000.csv')

# Analyze results
print(f"Best score: {study_data['best_value']}")
print(f"Best parameters: {study_data['best_params']}")
print(f"Top 5 trials:")
print(df.nlargest(5, 'score')[['trial_id', 'score', 'duration_seconds']])
```

## Best Practices

1. **Start with quick test** to verify everything works
2. **Use standard config** for development
3. **Use comprehensive config** for research
4. **Use production config** for final optimization
5. **Monitor progress** during long runs
6. **Save results** for analysis

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or hidden dimensions
2. **Slow training**: Reduce epochs or use fewer trials
3. **No improvement**: Check learning rate range
4. **Crashes**: Check GPU availability and memory

### Debug Mode
```bash
python run_progression_hpo.py --config quick --study_name debug_test
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available
2. **Monitor memory**: Watch GPU memory usage
3. **Early stopping**: Use appropriate patience values
4. **Parallel trials**: Run multiple studies in parallel (different GPU IDs)

## Example Workflow

```bash
# 1. Quick test
python run_progression_hpo.py --config quick

# 2. Monitor progress
python monitor_hpo.py --watch

# 3. Standard optimization
python run_progression_hpo.py --config standard

# 4. Analyze results
python -c "
import json
with open('results/optuna_study_progression_hpo_*.json', 'r') as f:
    data = json.load(f)
print('Best score:', data['best_value'])
print('Best params:', data['best_params'])
"
```



