import os
import sys
import subprocess
import optuna
import re
from datetime import datetime
import contextlib

# Set your results directory here (update as needed)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
# Create the directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a date string for the log file
date_str = datetime.now().strftime("%Y-%m-%d")
log_path = os.path.join(RESULTS_DIR, f"optuna_{date_str}.log")

print(f"[Optuna HPO] Logging output to: {log_path}")
with open(log_path, 'a') as log_file:
    log_file.write(f"[Optuna HPO] Logging output to: {log_path}\n")

# Path to your training script
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train_test_controller_classification.py")

def run_training_with_params(params):
    cmd = [
        "python", "-u", TRAIN_SCRIPT,
        "--dataset_name", "Lung",
        "--bs", str(params["bs"]),
        "--dropout", str(params["dropout"]),
        "--en", f"Lung_{date_str}",
        "--epoch", "200",
        "--factor", str(params["factor"]),
        "--fcl", str(params["fcl"]),
        "--gcn_h", str(params["gcn_h"]),
        "--gpu_id", "0",
        "--heads", str(params["heads"]),
        "--loss", "NegativeLogLikelihood",
        "--lr", str(params["lr"]),
        "--min_lr", str(params["min_lr"]),
        "--model", "GATV2",
        "--no-fold",
        "--num_of_ff_layers", str(params["num_of_ff_layers"]),
        "--num_of_gcn_layers", str(params["num_of_gcn_layers"]),
        "--patience", "10",
        "--unit", "month",
        "--weight_decay", str(params["weight_decay"])
    ]
    msg = f"\n[Optuna HPO] Running command: {' '.join(cmd)}"
    print(msg)
    with open(log_path, 'a') as log_file:
        log_file.write(msg + '\n')
    output_lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')  # Stream output live
        with open(log_path, 'a') as log_file:
            log_file.write(line)
        output_lines.append(line)
    process.wait()
    output = ''.join(output_lines)
    print(f"[Optuna HPO] Output for this trial finished.")
    with open(log_path, 'a') as log_file:
        log_file.write("[Optuna HPO] Output for this trial finished.\n")
    match = re.search(r"Average validation score: ([0-9.]+)", output)
    if match:
        return float(match.group(1))
    else:
        print("[Optuna HPO] Could not find validation score in output.")
        with open(log_path, 'a') as log_file:
            log_file.write("[Optuna HPO] Could not find validation score in output.\n")
        return 0.0

def objective(trial):
    msg = f"\n[Optuna HPO] Starting trial {trial.number}..."
    print(msg)
    with open(log_path, 'a') as log_file:
        log_file.write(msg + '\n')
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "bs": trial.suggest_categorical("bs", [16, 32, 64]),
        "gcn_h": trial.suggest_categorical("gcn_h", [16, 32, 64, 128]),
        "fcl": trial.suggest_categorical("fcl", [64, 128, 256]),
        "num_of_gcn_layers": trial.suggest_int("num_of_gcn_layers", 2, 3),
        "num_of_ff_layers": trial.suggest_int("num_of_ff_layers", 1, 2),
        "heads": trial.suggest_categorical("heads", [1, 2]),
        "factor": trial.suggest_float("factor", 0.1, 0.5),
        "min_lr": trial.suggest_float("min_lr", 1e-5, 1e-3, log=True),
    }
    msg = f"[Optuna HPO] Trial {trial.number} parameters: {params}"
    print(msg)
    with open(log_path, 'a') as log_file:
        log_file.write(msg + '\n')
    mean_val_score = run_training_with_params(params)
    msg = f"[Optuna HPO] Finished trial {trial.number} with score: {mean_val_score}\n"
    print(msg)
    with open(log_path, 'a') as log_file:
        log_file.write(msg + '\n')
    return mean_val_score

if __name__ == "__main__":
    msg = f"[Optuna HPO] Logging output to: {log_path}"
    with open(log_path, 'a') as log_file:
        log_file.write(msg + '\n')
    print(msg)
    msg = "[Optuna HPO] Starting hyperparameter optimization..."
    print(msg)
    with open(log_path, 'a') as log_file:
        log_file.write(msg + '\n')
    with open(log_path, 'a') as log_file, contextlib.redirect_stdout(log_file):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1000)
        msg = "[Optuna HPO] Best trial:"
        print(msg)
        log_file.write(msg + '\n')
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        log_file.write(f"  Value: {trial.value}\n")
        log_file.write("  Params: \n")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            log_file.write(f"    {key}: {value}\n")