#!/usr/bin/env python3
"""
Launcher script for Progression hyperparameter optimization.
Provides easy access to different optimization scenarios.
"""

import sys
import argparse
from pathlib import Path
from hpo_config import get_config, print_available_configs

def main():
    parser = argparse.ArgumentParser(
        description="Progression Classification Hyperparameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_progression_hpo.py --config quick          # Quick test (3 trials)
  python run_progression_hpo.py --config standard      # Standard optimization (50 trials)
  python run_progression_hpo.py --config comprehensive # Comprehensive (100 trials)
  python run_progression_hpo.py --config production     # Production (200 trials)
  python run_progression_hpo.py --custom --trials 20   # Custom number of trials
        """
    )
    
    parser.add_argument(
        '--config', 
        choices=['quick', 'standard', 'comprehensive', 'production'],
        default='standard',
        help='Optimization configuration to use'
    )
    
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Use custom parameters instead of predefined config'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of trials (only used with --custom)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of epochs per trial (only used with --custom)'
    )
    
    parser.add_argument(
        '--study_name',
        type=str,
        help='Custom study name'
    )
    
    parser.add_argument(
        '--list_configs',
        action='store_true',
        help='List available configurations and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_configs:
        print_available_configs()
        return
    
    # Import the HPO class
    from optuna_hpo_progression_full import ProgressionHPO
    
    if args.custom:
        # Custom configuration
        study_name = args.study_name or f"progression_custom_{args.trials}t"
        hpo = ProgressionHPO(n_trials=args.trials, study_name=study_name, epochs=args.epochs)
        print(f"Running custom optimization: {args.trials} trials, {args.epochs} epochs")
    else:
        # Predefined configuration
        config = get_config(args.config)
        study_name = args.study_name or config['study_name']
        hpo = ProgressionHPO(n_trials=config['n_trials'], study_name=study_name, epochs=config['epochs'])
        print(f"Running {args.config} optimization: {config['n_trials']} trials, {config['epochs']} epochs")
    
    # Run optimization
    try:
        study = hpo.run_optimization()
        print(f"\nOptimization completed successfully!")
        print(f"Best score: {study.best_value:.6f}")
        print(f"Best trial: {study.best_trial.number}")
        return 0
    except KeyboardInterrupt:
        print(f"\nOptimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\nOptimization failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())



