"""
Configuration file for hyperparameter optimization scenarios.
"""

# Quick test configuration (for testing)
QUICK_TEST_CONFIG = {
    "n_trials": 3,
    "epochs": 5,
    "patience": 5,
    "study_name": "progression_quick_test",
    "models": ["GATV2"],
    "batch_sizes": [16, 32, 64],
    "learning_rates": {"min": 1e-4, "max": 1e-2},
    "hidden_sizes": [32, 64, 128],
    "num_layers": {"gcn": [2, 4], "ff": [1, 3]}
}

# Standard optimization configuration
STANDARD_CONFIG = {
    "n_trials": 50,
    "epochs": 200,
    "patience": 10,
    "study_name": "progression_standard",
    "models": ["GATV2"],
    "batch_sizes": [16, 32, 64],
    "learning_rates": {"min": 1e-5, "max": 1e-2},
    "hidden_sizes": [32, 64, 128, 256],
    "num_layers": {"gcn": [1, 4], "ff": [1, 3]}
}

# Comprehensive optimization configuration
COMPREHENSIVE_CONFIG = {
    "n_trials": 100,
    "epochs": 300,
    "patience": 15,
    "study_name": "progression_comprehensive",
    "models": ["GATV2"],
    "batch_sizes": [8, 16, 32, 64, 128],
    "learning_rates": {"min": 1e-6, "max": 1e-2},
    "hidden_sizes": [16, 32, 64, 128, 256, 512],
    "num_layers": {"gcn": [1, 5], "ff": [1, 4]}
}

# Production configuration (for final optimization)
PRODUCTION_CONFIG = {
    "n_trials": 1000,
    "epochs": 100,
    "patience": 10,
    "study_name": "progression_production",
    "models": ["GATV2"],
    "batch_sizes": [8, 16, 32, 64, 128],
    "learning_rates": {"min": 1e-6, "max": 1e-2},
    "hidden_sizes": [16, 32, 64, 128, 256, 512],
    "num_layers": {"gcn": [1, 6], "ff": [1, 5]}
}

# Configuration mapping
CONFIGS = {
    "quick": QUICK_TEST_CONFIG,
    "standard": STANDARD_CONFIG,
    "comprehensive": COMPREHENSIVE_CONFIG,
    "production": PRODUCTION_CONFIG
}

def get_config(config_name="standard"):
    """Get configuration by name."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]

def print_available_configs():
    """Print available configurations."""
    print("Available HPO configurations:")
    for name, config in CONFIGS.items():
        print(f"  {name}: {config['n_trials']} trials, {config['epochs']} epochs, {len(config['models'])} models")
