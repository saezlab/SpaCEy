#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Optimization for Progression Classification
Using Optuna for automated hyperparameter tuning with extensive search space.
"""

import os
import sys
import subprocess
import optuna
import re
import json
import pandas as pd
from datetime import datetime
import contextlib
import random
import string
import numpy as np
from pathlib import Path

def generate_random_string(length=10):
    """Generate random string for unique log file names."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

class ProgressionHPO:
    def __init__(self, n_trials=100, study_name="progression_hpo", epochs=100):
        """Initialize the hyperparameter optimization."""
        self.n_trials = n_trials
        self.study_name = study_name
        self.epochs = epochs
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Create unique log file
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_path = self.results_dir / f"optuna_progression_full_{date_str}_{generate_random_string()}.log"
        
        # Initialize logging
        self._setup_logging()
        
        # Training script path
        self.train_script = Path(__file__).parent / "train_test_controller_classification.py"
        
        # Results tracking
        self.trial_results = []
        self.best_score = 0.0
        self.best_params = None
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.log(f"[Progression HPO] Starting comprehensive hyperparameter optimization")
        self.log(f"[Progression HPO] Log file: {self.log_path}")
        self.log(f"[Progression HPO] Number of trials: {self.n_trials}")
        self.log(f"[Progression HPO] Study name: {self.study_name}")
        
    def log(self, message):
        """Log message to both console and file."""
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(f"{message}\n")
    
    def suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for the trial."""
        params = {
            # Learning rate and optimization - more conservative ranges
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "min_lr": trial.suggest_float("min_lr", 1e-5, 1e-4, log=True),
            
            # Model architecture - more focused ranges
            "gcn_h": trial.suggest_categorical("gcn_h", [32, 64, 128]),
            "fcl": trial.suggest_categorical("fcl", [64, 128, 256]),
            "num_of_gcn_layers": trial.suggest_int("num_of_gcn_layers", 2, 3),
            "num_of_ff_layers": trial.suggest_int("num_of_ff_layers", 1, 2),
            "heads": trial.suggest_categorical("heads", [1, 2]),
            
            # Regularization - more conservative
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            
            # Training configuration - faster training
            "bs": trial.suggest_categorical("bs", [16, 32]),
            "patience": trial.suggest_int("patience", 5, 10),
            
            # Learning rate scheduling
            "factor": trial.suggest_float("factor", 0.3, 0.7),
            
            # Model type - only GATV2
            "model": "GATV2",
        }
        
        return params
    
    def run_training_trial(self, params):
        """Run a single training trial with given parameters."""
        cmd = [
            "python", str(self.train_script),
            "--dataset_name", "Lung",
            "--label", "Progression",
            "--bs", str(params["bs"]),
            "--dropout", str(params["dropout"]),
            "--en", f"progression_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--epoch", str(self.epochs),  # Use configurable epochs
            "--factor", str(params["factor"]),
            "--fcl", str(params["fcl"]),
            "--gcn_h", str(params["gcn_h"]),
            "--gpu_id", "0",
            "--heads", str(params["heads"]),
            "--loss", "BCEWithLogitsLoss",
            "--lr", str(params["lr"]),
            "--min_lr", str(params["min_lr"]),
            "--model", params["model"],
            "--fold",
            "--no-full_training",
            "--no-t_v_t",
            "--num_of_ff_layers", str(params["num_of_ff_layers"]),
            "--num_of_gcn_layers", str(params["num_of_gcn_layers"]),
            "--patience", str(params["patience"]),
            "--unit", "month",
            "--weight_decay", str(params["weight_decay"])
        ]
        
        self.log(f"[Trial] Running command: {' '.join(cmd)}")
        
        # Run training with timeout
        output_lines = []
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1
            )
            
            # Set timeout (20 minutes per trial)
            timeout_seconds = 1200
            start_time = datetime.now()
            
            for line in process.stdout:
                output_lines.append(line)
                # Log important lines
                if any(keyword in line.lower() for keyword in ["epoch", "auc", "accuracy", "loss", "best", "early"]):
                    self.log(f"[Training] {line.strip()}")
                
                # Check timeout
                if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                    self.log(f"[TIMEOUT] Trial exceeded {timeout_seconds} seconds, terminating...")
                    process.terminate()
                    return 0.0
            
            process.wait(timeout=timeout_seconds)
            
            if process.returncode != 0:
                self.log(f"[ERROR] Training failed with return code {process.returncode}")
                return 0.0
                
        except subprocess.TimeoutExpired:
            self.log(f"[TIMEOUT] Training exceeded timeout, terminating...")
            process.kill()
            return 0.0
        except Exception as e:
            self.log(f"[ERROR] Exception during training: {str(e)}")
            return 0.0
        
        # Parse results
        output = ''.join(output_lines)
        return self._parse_training_results(output)
    
    def _parse_training_results(self, output):
        """Parse training results from output."""
        # Try to find validation score with multiple patterns
        patterns = [
            r"Average validation score: ([0-9.]+)",
            r"Best validation score: ([0-9.]+)",
            r"Validation AUC: ([0-9.]+)",
            r"Val AUC: ([0-9.]+)",
            r"Best val\. score: ([0-9.]+)",
            r"Best eval score: ([0-9.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                score = float(match.group(1))
                self.log(f"[Result] Found validation score: {score}")
                return score
        
        # If no specific score found, try to extract from training logs
        auc_matches = re.findall(r"Val AUC: ([0-9.]+)", output)
        if auc_matches:
            # Take the last (best) AUC score
            score = float(auc_matches[-1])
            self.log(f"[Result] Using last AUC score: {score}")
            return score
        
        # Try to find any AUC score in the output
        general_auc_matches = re.findall(r"AUC: ([0-9.]+)", output)
        if general_auc_matches:
            score = float(general_auc_matches[-1])
            self.log(f"[Result] Using general AUC score: {score}")
            return score
        
        # Try to find any score pattern
        score_patterns = [
            r"score: ([0-9.]+)",
            r"Score: ([0-9.]+)",
            r"validation.*?([0-9.]+)",
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    scores = [float(m) for m in matches]
                    # Take the highest score found
                    score = max(scores)
                    self.log(f"[Result] Using max score from pattern: {score}")
                    return score
                except ValueError:
                    continue
        
        self.log(f"[WARNING] Could not find validation score in output")
        self.log(f"[DEBUG] Output sample: {output[-1000:]}")  # Show last 1000 chars for debugging
        return 0.0
    
    def objective(self, trial):
        """Optuna objective function."""
        trial_id = trial.number
        self.log(f"\n{'='*60}")
        self.log(f"[Trial {trial_id}] Starting hyperparameter optimization trial")
        self.log(f"{'='*60}")
        
        # Suggest hyperparameters
        params = self.suggest_hyperparameters(trial)
        self.log(f"[Trial {trial_id}] Parameters: {json.dumps(params, indent=2)}")
        
        # Run training
        start_time = datetime.now()
        score = self.run_training_trial(params)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log results
        self.log(f"[Trial {trial_id}] Completed in {duration:.1f} seconds")
        self.log(f"[Trial {trial_id}] Validation score: {score}")
        
        # Track results
        trial_result = {
            'trial_id': trial_id,
            'score': score,
            'duration_seconds': duration,
            'params': params,
            'timestamp': start_time.isoformat()
        }
        self.trial_results.append(trial_result)
        
        # Update best results
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            self.log(f"[Trial {trial_id}] NEW BEST SCORE: {score}")
        
        return score
    
    def run_optimization(self):
        """Run the complete hyperparameter optimization."""
        self.log(f"\n{'='*80}")
        self.log(f"STARTING COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
        self.log(f"Study: {self.study_name}")
        self.log(f"Trials: {self.n_trials}")
        self.log(f"{'='*80}")
        
        try:
            # Create Optuna study
            study = optuna.create_study(
                direction="maximize",
                study_name=self.study_name,
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            
            # Run optimization
            study.optimize(self.objective, n_trials=self.n_trials)
            
            # Save results
            self._save_results(study)
            
            return study
            
        except Exception as e:
            self.log(f"[ERROR] Optimization failed: {str(e)}")
            raise
    
    def _save_results(self, study):
        """Save optimization results."""
        # Save study results
        study_path = self.results_dir / f"optuna_study_{self.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert study to JSON-serializable format
        study_data = {
            'study_name': study.study_name,
            'n_trials': len(study.trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial_number': study.best_trial.number,
            'trials': []
        }
        
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            }
            study_data['trials'].append(trial_data)
        
        with open(study_path, 'w') as f:
            json.dump(study_data, f, indent=2)
        
        self.log(f"[Results] Study saved to: {study_path}")
        
        # Save trial results as CSV
        if self.trial_results:
            df = pd.DataFrame(self.trial_results)
            csv_path = self.results_dir / f"trial_results_{self.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            self.log(f"[Results] Trial results saved to: {csv_path}")
        
        # Print summary
        self._print_summary(study)
    
    def _print_summary(self, study):
        """Print optimization summary."""
        self.log(f"\n{'='*80}")
        self.log(f"OPTIMIZATION SUMMARY")
        self.log(f"{'='*80}")
        self.log(f"Study name: {study.study_name}")
        self.log(f"Total trials: {len(study.trials)}")
        self.log(f"Best score: {study.best_value:.6f}")
        self.log(f"Best trial: {study.best_trial.number}")
        self.log(f"Best parameters:")
        for key, value in study.best_params.items():
            self.log(f"  {key}: {value}")
        
        # Top 5 trials
        sorted_trials = sorted(study.trials, key=lambda x: x.value or 0, reverse=True)[:5]
        self.log(f"\nTop 5 trials:")
        for i, trial in enumerate(sorted_trials, 1):
            self.log(f"  {i}. Trial {trial.number}: {trial.value:.6f}")
        
        self.log(f"{'='*80}")

def main():
    """Main function to run hyperparameter optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Progression Classification HPO')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials to run')
    parser.add_argument('--study_name', type=str, default='progression_hpo', help='Study name')
    parser.add_argument('--quick_test', action='store_true', help='Run quick test with 3 trials')
    
    args = parser.parse_args()
    
    if args.quick_test:
        n_trials = 3
        study_name = "progression_hpo_test"
    else:
        n_trials = args.n_trials
        study_name = args.study_name
    
    # Create and run optimization
    hpo = ProgressionHPO(n_trials=n_trials, study_name=study_name)
    study = hpo.run_optimization()
    
    return study

if __name__ == "__main__":
    study = main()
