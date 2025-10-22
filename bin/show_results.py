#!/usr/bin/env python3
"""
Script to display scalability simulation results in a readable format.
"""

import pandas as pd
import sys
import os
from pathlib import Path

def show_results(results_dir):
    """Display the results from the scalability simulation."""
    
    # Find the most recent results directory
    if results_dir is None:
        results_path = Path("../results")
        if not results_path.exists():
            print("No results directory found!")
            return
        
        # Find the most recent scalability simulation
        sim_dirs = [d for d in results_path.iterdir() if d.is_dir() and "scalability_simulation" in d.name]
        if not sim_dirs:
            print("No scalability simulation results found!")
            return
        
        # Sort by creation time and get the most recent
        sim_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        results_dir = sim_dirs[0]
    
    csv_file = Path(results_dir) / "data" / "simulation_results.csv"
    
    if not csv_file.exists():
        print(f"Results file not found: {csv_file}")
        return
    
    # Load the results
    df = pd.read_csv(csv_file)
    
    print("=" * 80)
    print("GNN RUNTIME SCALABILITY SIMULATION RESULTS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Total configurations tested: {len(df)}")
    print()
    
    # Show marker, sample, and node counts
    marker_counts = sorted(df['num_markers'].unique())
    sample_counts = sorted(df['num_samples'].unique())
    node_counts = sorted(df['num_nodes'].unique())
    
    print(f"Marker counts tested: {marker_counts}")
    print(f"Sample counts tested: {sample_counts}")
    print(f"Node counts tested: {node_counts}")
    print()
    
    # Create summary table
    print("RUNTIME SUMMARY TABLE")
    print("-" * 100)
    print(f"{'Markers':<8} {'Samples':<8} {'Nodes':<8} {'Total Time (s)':<15} {'Avg Epoch (s)':<15} {'Peak Memory (MB)':<15}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['num_markers']:<8} {row['num_samples']:<8} {row['num_nodes']:<8} {row['total_training_time']:<15.2f} {row['avg_epoch_time']:<15.3f} {row['peak_memory_usage']:<15.1f}")
    
    print("-" * 80)
    print()
    
    # Show fastest and slowest configurations
    fastest = df.loc[df['total_training_time'].idxmin()]
    slowest = df.loc[df['total_training_time'].idxmax()]
    
    print("PERFORMANCE EXTREMES")
    print("-" * 40)
    print(f"Fastest configuration:")
    print(f"  {fastest['num_markers']} markers, {fastest['num_samples']} samples, {fastest['num_nodes']} nodes")
    print(f"  Total time: {fastest['total_training_time']:.2f}s")
    print(f"  Average epoch: {fastest['avg_epoch_time']:.3f}s")
    print(f"  Peak memory: {fastest['peak_memory_usage']:.1f}MB")
    print()
    
    print(f"Slowest configuration:")
    print(f"  {slowest['num_markers']} markers, {slowest['num_samples']} samples, {slowest['num_nodes']} nodes")
    print(f"  Total time: {slowest['total_training_time']:.2f}s")
    print(f"  Average epoch: {slowest['avg_epoch_time']:.3f}s")
    print(f"  Peak memory: {slowest['peak_memory_usage']:.1f}MB")
    print()
    
    # Show scaling analysis
    print("SCALING ANALYSIS")
    print("-" * 40)
    
    # Time scaling with nodes (averaged over samples and markers)
    print("Time scaling with number of nodes (averaged over samples and markers):")
    node_avg = df.groupby('num_nodes')['total_training_time'].mean().sort_index()
    for nodes, avg_time in node_avg.items():
        print(f"  {nodes:>5} nodes: {avg_time:>6.2f}s")
    
    print()
    
    # Time scaling with samples (for a fixed number of markers and nodes)
    print("Time scaling with number of samples (for 1000 markers, 1000 nodes):")
    subset_1000 = df[(df['num_markers'] == 1000) & (df['num_nodes'] == 1000)].sort_values('num_samples')
    for _, row in subset_1000.iterrows():
        print(f"  {row['num_samples']:>5} samples: {row['total_training_time']:>6.2f}s")
    
    print()
    
    # Time scaling with markers (for a fixed number of samples and nodes)
    print("Time scaling with number of markers (for 1000 samples, 1000 nodes):")
    subset_1000_samples = df[(df['num_samples'] == 1000) & (df['num_nodes'] == 1000)].sort_values('num_markers')
    for _, row in subset_1000_samples.iterrows():
        print(f"  {row['num_markers']:>5} markers: {row['total_training_time']:>6.2f}s")
    
    print()
    
    # Memory analysis
    print("MEMORY ANALYSIS")
    print("-" * 40)
    max_memory = df['peak_memory_usage'].max()
    min_memory = df['peak_memory_usage'].min()
    avg_memory = df['peak_memory_usage'].mean()
    
    print(f"Peak memory usage range: {min_memory:.1f}MB - {max_memory:.1f}MB")
    print(f"Average peak memory: {avg_memory:.1f}MB")
    print()
    
    # Show plots location
    plots_dir = Path(results_dir) / "plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print("GENERATED PLOTS")
        print("-" * 40)
        for plot_file in plot_files:
            print(f"  {plot_file.name}")
        print(f"\nPlots location: {plots_dir}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Check if a specific results directory was provided
    results_dir = None
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    show_results(results_dir)
