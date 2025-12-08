#!/usr/bin/env python3
"""
Test script for improved hyperparameter optimization.
"""

import sys
from pathlib import Path

# Add the bin directory to the path
sys.path.append(str(Path(__file__).parent))

from optuna_hpo_progression_full import ProgressionHPO

def test_improved_hpo():
    """Test the improved HPO implementation."""
    print("Testing improved hyperparameter optimization...")
    
    # Create HPO instance with minimal trials
    hpo = ProgressionHPO(n_trials=2, study_name="test_improved")
    
    # Test hyperparameter suggestion
    import optuna
    study = optuna.create_study(direction="maximize")
    
    def test_objective(trial):
        return hpo.objective(trial)
    
    # Run a few trials
    study.optimize(test_objective, n_trials=2)
    
    print(f"Test completed. Best score: {study.best_value}")
    print(f"Best parameters: {study.best_params}")

if __name__ == "__main__":
    test_improved_hpo()



