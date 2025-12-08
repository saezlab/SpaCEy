#!/usr/bin/env python3
"""
Monitor hyperparameter optimization progress.
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def find_latest_log(results_dir):
    """Find the latest optimization log file."""
    results_path = Path(results_dir)
    log_files = list(results_path.glob("optuna_progression_full_*.log"))
    if not log_files:
        return None
    
    # Sort by modification time, get the latest
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    return latest_log

def parse_log_file(log_path):
    """Parse log file to extract trial information."""
    trials = []
    current_trial = None
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Trial start
            if "Starting trial" in line:
                trial_match = re.search(r"Trial (\d+)", line)
                if trial_match:
                    current_trial = {
                        'trial_id': int(trial_match.group(1)),
                        'start_time': None,
                        'end_time': None,
                        'score': None,
                        'duration': None,
                        'status': 'running'
                    }
            
            # Trial parameters
            elif "Parameters:" in line and current_trial:
                # Extract parameters from JSON-like format
                try:
                    params_start = line.find('{')
                    if params_start != -1:
                        params_str = line[params_start:]
                        params = json.loads(params_str)
                        current_trial['params'] = params
                except:
                    pass
            
            # Trial completion
            elif "Completed in" in line and current_trial:
                duration_match = re.search(r"(\d+\.\d+) seconds", line)
                if duration_match:
                    current_trial['duration'] = float(duration_match.group(1))
                    current_trial['status'] = 'completed'
            
            # Validation score
            elif "Validation score:" in line and current_trial:
                score_match = re.search(r"Validation score: ([\d.]+)", line)
                if score_match:
                    current_trial['score'] = float(score_match.group(1))
            
            # Best score update
            elif "NEW BEST SCORE:" in line and current_trial:
                best_match = re.search(r"NEW BEST SCORE: ([\d.]+)", line)
                if best_match:
                    current_trial['is_best'] = True
            
            # Trial finished
            elif "Finished trial" in line and current_trial:
                trials.append(current_trial)
                current_trial = None
    
    return trials

def print_progress(trials):
    """Print optimization progress."""
    if not trials:
        print("No trials found in log file.")
        return
    
    completed_trials = [t for t in trials if t.get('status') == 'completed']
    running_trials = [t for t in trials if t.get('status') == 'running']
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER OPTIMIZATION PROGRESS")
    print(f"{'='*60}")
    print(f"Total trials: {len(trials)}")
    print(f"Completed: {len(completed_trials)}")
    print(f"Running: {len(running_trials)}")
    
    if completed_trials:
        scores = [t['score'] for t in completed_trials if t.get('score') is not None]
        if scores:
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            print(f"Best score: {best_score:.6f}")
            print(f"Average score: {avg_score:.6f}")
            
            # Show top 3 trials
            sorted_trials = sorted(completed_trials, key=lambda x: x.get('score', 0), reverse=True)[:3]
            print(f"\nTop 3 trials:")
            for i, trial in enumerate(sorted_trials, 1):
                score = trial.get('score', 0)
                trial_id = trial.get('trial_id', 'N/A')
                duration = trial.get('duration', 0)
                print(f"  {i}. Trial {trial_id}: {score:.6f} (duration: {duration:.1f}s)")
    
    if running_trials:
        print(f"\nCurrently running trials:")
        for trial in running_trials:
            trial_id = trial.get('trial_id', 'N/A')
            print(f"  Trial {trial_id}")

def print_best_params(trials):
    """Print best parameters found so far."""
    completed_trials = [t for t in trials if t.get('status') == 'completed' and t.get('score') is not None]
    if not completed_trials:
        print("No completed trials with scores found.")
        return
    
    best_trial = max(completed_trials, key=lambda x: x.get('score', 0))
    
    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS SO FAR")
    print(f"{'='*60}")
    print(f"Trial ID: {best_trial.get('trial_id', 'N/A')}")
    print(f"Score: {best_trial.get('score', 0):.6f}")
    print(f"Duration: {best_trial.get('duration', 0):.1f} seconds")
    
    if 'params' in best_trial:
        print(f"\nParameters:")
        for key, value in best_trial['params'].items():
            print(f"  {key}: {value}")

def save_progress_csv(trials, output_path):
    """Save trial progress to CSV."""
    if not trials:
        print("No trials to save.")
        return
    
    # Flatten trial data for CSV
    csv_data = []
    for trial in trials:
        row = {
            'trial_id': trial.get('trial_id'),
            'status': trial.get('status'),
            'score': trial.get('score'),
            'duration': trial.get('duration'),
            'is_best': trial.get('is_best', False)
        }
        
        # Add parameters as columns
        if 'params' in trial:
            for key, value in trial['params'].items():
                row[f'param_{key}'] = value
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    print(f"Progress saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Monitor HPO progress")
    parser.add_argument('--log_file', type=str, help='Specific log file to monitor')
    parser.add_argument('--results_dir', type=str, default='../results', help='Results directory')
    parser.add_argument('--save_csv', type=str, help='Save progress to CSV file')
    parser.add_argument('--watch', action='store_true', help='Watch mode (refresh every 30s)')
    
    args = parser.parse_args()
    
    # Find log file
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = find_latest_log(args.results_dir)
    
    if not log_path or not log_path.exists():
        print(f"No log file found in {args.results_dir}")
        return
    
    print(f"Monitoring: {log_path}")
    
    if args.watch:
        import time
        try:
            while True:
                os.system('clear')
                trials = parse_log_file(log_path)
                print_progress(trials)
                print_best_params(trials)
                print(f"\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("Press Ctrl+C to stop monitoring...")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        trials = parse_log_file(log_path)
        print_progress(trials)
        print_best_params(trials)
        
        if args.save_csv:
            save_progress_csv(trials, args.save_csv)

if __name__ == "__main__":
    main()



