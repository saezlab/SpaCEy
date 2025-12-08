#!/usr/bin/env python3
"""
Script to compare SOTA model results and create visualizations.
Collects metrics from Theis work, Space-GM, and ExpMap models.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_theis_results(filepath):
    """Load Theis work results."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = []
    # Extract fold-level metrics
    fold_accuracies = data['user_attrs']['fold_accuracies']
    fold_precisions = data['user_attrs']['fold_precisions']
    fold_recalls = data['user_attrs']['fold_recalls']
    fold_f1_scores = data['user_attrs']['fold_f1_scores']
    fold_aucs = data['user_attrs']['fold_aucs']
    fold_auprcs = data['user_attrs']['fold_auprcs']
    
    for i in range(len(fold_accuracies)):
        results.append({
            'model': 'Theis',
            'fold': i,
            'accuracy': fold_accuracies[i],
            'precision': fold_precisions[i],
            'recall': fold_recalls[i],
            'f1': fold_f1_scores[i],
            'auc': fold_aucs[i],
            'auprc': fold_auprcs[i]
        })
    
    return results

def load_spacegm_results(filepath):
    """Load Space-GM results."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = []
    # Extract fold-level metrics
    for fold_data in data['best_fold_results']:
        results.append({
            'model': 'Space-GM',
            'fold': fold_data['fold'],
            'accuracy': fold_data['accuracy'],
            'precision': fold_data['precision'],
            'recall': fold_data['recall'],
            'f1': fold_data['f1'],
            'auc': fold_data['auc'],
            'auprc': fold_data['auprc']
        })
    
    return results

def load_expmap_results(csv_filepath):
    """Load ExpMap results from per_fold_results CSV file."""
    df = pd.read_csv(csv_filepath)
    
    results = []
    for _, row in df.iterrows():
        results.append({
            'model': 'ExpMap',
            'fold': int(row['Fold']),
            'accuracy': row['Best Val Accuracy'],
            'precision': row['Best Val Precision'],
            'recall': row['Best Val Recall'],
            'f1': row['Best Val F1'],
            'auc': row['Best Val AUC'],
            'auprc': row['Best Val AUPRC']
        })
    
    return results

def create_comparison_csv(results_list, output_path):
    """Create a CSV file with all metrics."""
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"CSV file saved to: {output_path}")
    return df

def create_boxplots(df, output_dir):
    """Create box plots for different metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
    
    # Filter out None values for each metric
    for metric in metrics:
        metric_df = df[['model', 'fold', metric]].copy()
        metric_df = metric_df[metric_df[metric].notna()]
        
        if len(metric_df) == 0:
            print(f"Skipping {metric} - no data available")
            continue
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metric_df, x='model', y=metric, hue='model', palette='Set2', legend=False)
        plt.title(f'{metric.upper()} Comparison Across Models', fontsize=14, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # Save as both PNG and PDF
        png_path = output_dir / f'{metric}_comparison_boxplot.png'
        pdf_path = output_dir / f'{metric}_comparison_boxplot.pdf'
        plt.savefig(png_path, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved {metric} boxplot to {png_path} and {pdf_path}")
        plt.close()
    
    # Create a combined plot with all metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        metric_df = df[['model', 'fold', metric]].copy()
        metric_df = metric_df[metric_df[metric].notna()]
        
        if len(metric_df) == 0:
            axes[idx].text(0.5, 0.5, f'{metric}\nNo data', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            continue
        
        sns.boxplot(data=metric_df, x='model', y=metric, hue='model', palette='Set2', legend=False, ax=axes[idx])
        axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Model', fontsize=10)
        axes[idx].set_ylabel(metric.upper(), fontsize=10)
        axes[idx].tick_params(axis='x', rotation=0)
    
    plt.suptitle('Model Comparison: All Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    combined_png = output_dir / 'all_metrics_comparison_boxplot.png'
    combined_pdf = output_dir / 'all_metrics_comparison_boxplot.pdf'
    plt.savefig(combined_png, bbox_inches='tight')
    plt.savefig(combined_pdf, bbox_inches='tight')
    print(f"Saved combined boxplot to {combined_png} and {combined_pdf}")
    plt.close()

def create_barplots(df, output_dir):
    """Create bar plots with error bars (mean ± std) for different metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']
    
    # Define model order: Space-GM, Theis, ExpMap
    model_order = ['Space-GM', 'Theis', 'ExpMap']
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Matching colors for each model
    
    # Calculate summary statistics for each model and metric
    summary_stats = {}
    for model in model_order:
        summary_stats[model] = {}
        model_df = df[df['model'] == model]
        for metric in metrics:
            metric_values = model_df[metric].dropna()
            if len(metric_values) > 0:
                summary_stats[model][metric] = {
                    'mean': metric_values.mean(),
                    'std': metric_values.std()
                }
            else:
                summary_stats[model][metric] = {
                    'mean': np.nan,
                    'std': np.nan
                }
    
    # Create individual bar plots for each metric
    for metric in metrics:
        means = []
        stds = []
        valid_models = []
        valid_colors = []
        
        for i, model in enumerate(model_order):
            if metric in summary_stats[model] and not np.isnan(summary_stats[model][metric]['mean']):
                means.append(summary_stats[model][metric]['mean'])
                stds.append(summary_stats[model][metric]['std'])
                valid_models.append(model)
                valid_colors.append(colors[i])
        
        if len(means) == 0:
            print(f"Skipping {metric} - no data available")
            continue
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(valid_models))
        bars = plt.bar(x_pos, means, yerr=stds, capsize=5, color=valid_colors, 
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.ylabel(metric.upper(), fontsize=12, fontweight='bold')
        plt.title(f'{metric.upper()} Comparison Across Models (Mean ± Std)', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, valid_models, fontsize=11)
        plt.ylim(0.30, 1.00)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save as both PNG and PDF
        png_path = output_dir / f'{metric}_comparison_barplot.png'
        pdf_path = output_dir / f'{metric}_comparison_barplot.pdf'
        plt.savefig(png_path, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved {metric} barplot to {png_path} and {pdf_path}")
        plt.close()
    
    # Create a single combined plot with all metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        means = []
        stds = []
        valid_models = []
        valid_colors = []
        
        # Maintain order but only include models with data
        for i, model in enumerate(model_order):
            if metric in summary_stats[model] and not np.isnan(summary_stats[model][metric]['mean']):
                means.append(summary_stats[model][metric]['mean'])
                stds.append(summary_stats[model][metric]['std'])
                valid_models.append(model)
                valid_colors.append(colors[i])
        
        if len(means) == 0:
            axes[idx].text(0.5, 0.5, f'{metric}\nNo data', 
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=12)
            axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
            continue
        
        x_pos = np.arange(len(valid_models))
        bars = axes[idx].bar(x_pos, means, yerr=stds, capsize=5, color=valid_colors,
                            alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Model', fontsize=10)
        axes[idx].set_ylabel(metric.upper(), fontsize=10)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(valid_models, fontsize=9)
        axes[idx].set_ylim(0.30, 1.00)
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Model Comparison: All Metrics (Mean ± Std)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    combined_png = output_dir / 'all_metrics_comparison_barplot.png'
    combined_pdf = output_dir / 'all_metrics_comparison_barplot.pdf'
    plt.savefig(combined_png, bbox_inches='tight')
    plt.savefig(combined_pdf, bbox_inches='tight')
    print(f"Saved combined barplot to {combined_png} and {combined_pdf}")
    plt.close()

def main():
    # File paths
    theis_path = Path('/home/rifaioglu/projects/tissue/hyperopt_full/best_trial.json')
    spacegm_path = Path('/home/rifaioglu/projects/space-gm/results/hyperparameter_optimization/quick_optimization_results.json')
    expmap_path = Path('/home/rifaioglu/projects/GNNClinicalOutcomePrediction/results/idedFiles/Lung_progression/progression_hpo_20251029_150511/nUb-AiGzK_Asp0_XZBa5pg_per_fold_results.csv')
    
    # Output directory
    output_dir = Path('/home/rifaioglu/projects/GNNClinicalOutcomePrediction/results/sota_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("Loading results...")
    theis_results = load_theis_results(theis_path)
    spacegm_results = load_spacegm_results(spacegm_path)
    expmap_results = load_expmap_results(expmap_path)
    
    print(f"Theis: {len(theis_results)} folds")
    print(f"Space-GM: {len(spacegm_results)} folds")
    print(f"ExpMap: {len(expmap_results)} folds")
    
    # Create CSV
    csv_path = output_dir / 'sota_comparison_results.csv'
    # df = create_comparison_csv([theis_results, spacegm_results, expmap_results], csv_path)
    df = pd.read_csv(csv_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    for model in df['model'].unique():
        print(f"\n{model}:")
        model_df = df[df['model'] == model]
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']:
            metric_values = model_df[metric].dropna()
            if len(metric_values) > 0:
                print(f"  {metric.upper()}: mean={metric_values.mean():.4f}, std={metric_values.std():.4f}, "
                      f"min={metric_values.min():.4f}, max={metric_values.max():.4f}")
    
    # Create box plots
    print("\nCreating box plots...")
    create_boxplots(df, output_dir)
    
    # Create bar plots with error bars
    print("\nCreating bar plots with error bars...")
    create_barplots(df, output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    main()

