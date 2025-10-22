#!/usr/bin/env python3
"""
Scalability Simulation Script for GNN Clinical Outcome Prediction

This script tests the scalability of the GNN method by measuring runtime
for different numbers of markers and samples.

Usage:
    python scalability_simulation.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import argparse
import psutil
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_tools import set_seeds, get_device, create_directories
from model import CustomGCN
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class ScalabilitySimulator:
    """Simulator for testing GNN method runtime scalability."""
    
    def __init__(self):
        """Initialize the simulator with fixed configuration."""
        self.device = get_device()
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Fixed configuration as requested
        self.marker_counts = [30, 50, 100, 500, 1000]
        self.sample_counts = [50, 100, 200, 500, 1000, 1500]
        self.node_distribution = {
            "mean": 1000,  # Mean number of nodes
            "std": 400,    # Standard deviation
            "min": 100,    # Minimum nodes (clipped)
            "max": 2000    # Maximum nodes (clipped)
        }
        self.epochs_per_config = 10
        
        # Model configuration
        self.model_config = {
            "model_type": "GATV2",
            "num_gcn_layers": 2,
            "num_ff_layers": 2,
            "gcn_hidden_neurons": 64,
            "ff_hidden_neurons": 128,
            "dropout": 0.2,
            "heads": 4
        }
        
        # Training configuration
        self.training_config = {
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "loss_function": "MSE"
        }
        
        # Synthetic data configuration
        self.synthetic_data_config = {
            "edge_probability": 0.1,
            "feature_noise": 0.1,
            "label_noise": 0.05
        }
        
        # Create output directories
        self.output_dir = Path(f"../results/scalability_simulation_{self.timestamp}")
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        create_directories([str(self.output_dir), str(self.plots_dir), str(self.data_dir)])
        
        print(f"Scalability simulation initialized. Output directory: {self.output_dir}")
        print(f"Testing {len(self.marker_counts)} marker counts: {self.marker_counts}")
        print(f"Testing {len(self.sample_counts)} sample counts: {self.sample_counts}")
        print(f"Testing node counts with normal distribution: mean={self.node_distribution['mean']}, std={self.node_distribution['std']}, range=[{self.node_distribution['min']}, {self.node_distribution['max']}]")
    
    def get_gpu_usage(self):
        """Get current GPU memory usage and utilization."""
        gpu_info = {}
        
        if torch.cuda.is_available():
            # Get GPU memory info
            gpu_info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            gpu_info['memory_reserved'] = torch.cuda.memory_reserved() / 1024**2    # MB
            gpu_info['memory_cached'] = torch.cuda.memory_cached() / 1024**2        # MB
            
            # Get total GPU memory
            gpu_info['total_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            
            # Calculate memory utilization percentage
            if gpu_info['total_memory'] > 0:
                gpu_info['memory_utilization_pct'] = (gpu_info['memory_allocated'] / gpu_info['total_memory']) * 100
            else:
                gpu_info['memory_utilization_pct'] = 0
            
            # Try to get GPU utilization if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info['gpu_utilization'] = util.gpu
                gpu_info['nvml_memory_utilization'] = util.memory
                pynvml.nvmlShutdown()
            except Exception as e:
                # If pynvml fails, set defaults
                gpu_info['gpu_utilization'] = 0
                gpu_info['nvml_memory_utilization'] = 0
        else:
            gpu_info['memory_allocated'] = 0
            gpu_info['memory_reserved'] = 0
            gpu_info['memory_cached'] = 0
            gpu_info['total_memory'] = 0
            gpu_info['memory_utilization_pct'] = 0
            gpu_info['gpu_utilization'] = 0
            gpu_info['nvml_memory_utilization'] = 0
        
        return gpu_info
    
    def generate_synthetic_dataset(self, num_markers, num_samples):
        """Generate synthetic dataset with specified number of markers and samples, with nodes following normal distribution."""
        print(f"Generating synthetic dataset: {num_markers} markers, {num_samples} samples, nodes ~N({self.node_distribution['mean']}, {self.node_distribution['std']})")
        
        data_list = []
        
        for sample_idx in tqdm(range(num_samples), desc="Generating samples"):
            # Generate number of nodes following normal distribution
            nodes_per_sample = int(np.random.normal(self.node_distribution['mean'], self.node_distribution['std']))
            # Clip to valid range
            nodes_per_sample = max(self.node_distribution['min'], min(self.node_distribution['max'], nodes_per_sample))
            
            # Node features (markers)
            x = torch.randn(nodes_per_sample, num_markers, dtype=torch.float32)
            
            # Generate random edges (sparse graph)
            edge_prob = self.synthetic_data_config["edge_probability"]
            max_edges = int(nodes_per_sample * (nodes_per_sample - 1) * edge_prob / 2)
            
            # Create random edges
            edges = []
            for _ in range(max_edges):
                src = np.random.randint(0, nodes_per_sample)
                dst = np.random.randint(0, nodes_per_sample)
                if src != dst:
                    edges.append([src, dst])
            
            if not edges:
                # If no edges, create a simple chain
                edges = [[i, i+1] for i in range(nodes_per_sample-1)]
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Generate synthetic labels (regression task)
            node_importance = torch.randn(num_markers)
            graph_feature = torch.mean(x, dim=0) @ node_importance
            structural_feature = edge_index.shape[1] / (nodes_per_sample * (nodes_per_sample - 1))
            
            # Add noise to make it realistic
            noise = torch.randn(1) * self.synthetic_data_config["label_noise"]
            y = torch.tensor([graph_feature + structural_feature + noise], dtype=torch.float32)
            
            # Generate coordinates (2D positions)
            pos = torch.randn(nodes_per_sample, 2, dtype=torch.float32)
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                pos=pos,
                y=y,
                # Add metadata for tracking
                sample_id=sample_idx,
                num_markers=num_markers,
                num_nodes=nodes_per_sample
            )
            
            data_list.append(data)
        
        return data_list
    
    def create_synthetic_dataset_class(self, data_list):
        """Create a dataset class from synthetic data."""
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
        
        return SyntheticDataset(data_list)
    
    def train_model(self, dataset, config_name):
        """Train the model for specified number of epochs and return runtime metrics."""
        print(f"Training model for configuration: {config_name}")
        
        # Create data loaders
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.training_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.training_config["batch_size"], shuffle=False)
        
        # Initialize model
        model = CustomGCN(
            type=self.model_config["model_type"],
            num_node_features=dataset[0].x.shape[1],
            num_gcn_layers=self.model_config["num_gcn_layers"],
            num_ff_layers=self.model_config["num_ff_layers"],
            gcn_hidden_neurons=self.model_config["gcn_hidden_neurons"],
            ff_hidden_neurons=self.model_config["ff_hidden_neurons"],
            dropout=self.model_config["dropout"],
            aggregators=["sum", "mean"],
            scalers=["identity", "amplification"],
            heads=self.model_config["heads"]
        ).to(self.device)
        
        # Setup training components
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"]
        )
        
        criterion = torch.nn.MSELoss()
        
        # Training loop
        train_times = []
        memory_usage = []
        gpu_usage_data = []
        
        model.train()
        
        for epoch in range(self.epochs_per_config):
            epoch_start_time = time.time()
            
            # Record GPU usage before training
            gpu_info = self.get_gpu_usage()
            gpu_usage_data.append(gpu_info)
            
            # Record memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            else:
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024**2)  # MB
            
            # Training
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y.squeeze())
                
                loss.backward()
                optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            train_times.append(epoch_time)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs_per_config}: Time: {epoch_time:.2f}s")
                print(f"    GPU Memory: {gpu_info['memory_allocated']:.1f}MB ({gpu_info['memory_utilization_pct']:.1f}% of {gpu_info['total_memory']:.0f}MB)")
                if gpu_info['gpu_utilization'] > 0:
                    print(f"    GPU Util: {gpu_info['gpu_utilization']}%")
        
        return {
            "config_name": config_name,
            "train_times": train_times,
            "memory_usage": memory_usage,
            "gpu_usage_data": gpu_usage_data,
            "avg_epoch_time": np.mean(train_times),
            "total_training_time": np.sum(train_times),
            "peak_memory_usage": np.max(memory_usage) if memory_usage else 0,
            "peak_gpu_memory": np.max([gpu['memory_allocated'] for gpu in gpu_usage_data]) if gpu_usage_data else 0,
            "avg_gpu_utilization": np.mean([gpu['gpu_utilization'] for gpu in gpu_usage_data]) if gpu_usage_data else 0,
            "avg_gpu_memory_utilization": np.mean([gpu['memory_utilization_pct'] for gpu in gpu_usage_data]) if gpu_usage_data else 0,
            "total_gpu_memory": gpu_usage_data[0]['total_memory'] if gpu_usage_data else 0
        }
    
    def run_simulation(self):
        """Run the complete scalability simulation."""
        print("Starting scalability simulation...")
        
        set_seeds(42)
        
        total_configs = len(self.marker_counts) * len(self.sample_counts)
        config_count = 0
        
        for num_markers in self.marker_counts:
            for num_samples in self.sample_counts:
                config_count += 1
                config_name = f"markers_{num_markers}_samples_{num_samples}"
                
                print(f"\n{'='*60}")
                print(f"Configuration {config_count}/{total_configs}: {config_name}")
                print(f"{'='*60}")
                
                # Generate synthetic dataset
                data_list = self.generate_synthetic_dataset(num_markers, num_samples)
                dataset = self.create_synthetic_dataset_class(data_list)
                
                # Train model and get metrics
                metrics = self.train_model(dataset, config_name)
                
                # Add configuration info
                metrics.update({
                    "num_markers": num_markers,
                    "num_samples": num_samples,
                    "min_nodes": min(data.num_nodes for data in data_list),
                    "max_nodes": max(data.num_nodes for data in data_list),
                    "avg_nodes_per_sample": np.mean([data.num_nodes for data in data_list]),
                    "total_nodes": sum(data.num_nodes for data in data_list)
                })
                
                self.results.append(metrics)
                
                # Save intermediate results
                self.save_results()
                
                # Clean up memory
                del dataset, data_list
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"\nSimulation completed! Results saved to {self.output_dir}")
        self.create_visualizations()
    
    def save_results(self):
        """Save results to JSON and CSV files."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = result.copy()
            serializable_result["train_times"] = result["train_times"]
            serializable_result["memory_usage"] = result["memory_usage"]
            # Don't serialize gpu_usage_data as it's too large
            if "gpu_usage_data" in serializable_result:
                del serializable_result["gpu_usage_data"]
            serializable_results.append(serializable_result)
        
        results_file = self.data_dir / "simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Also save as CSV for easy analysis
        df_results = self.create_results_dataframe()
        csv_file = self.data_dir / "simulation_results.csv"
        df_results.to_csv(csv_file, index=False)
    
    def create_results_dataframe(self):
        """Create a pandas DataFrame from results for easy analysis."""
        df_data = []
        
        for result in self.results:
            row = {
                "num_markers": result["num_markers"],
                "num_samples": result["num_samples"],
                "min_nodes": result["min_nodes"],
                "max_nodes": result["max_nodes"],
                "avg_nodes_per_sample": result["avg_nodes_per_sample"],
                "total_nodes": result["total_nodes"],
                "avg_epoch_time": result["avg_epoch_time"],
                "total_training_time": result["total_training_time"],
                "peak_memory_usage": result["peak_memory_usage"],
                "peak_gpu_memory": result.get("peak_gpu_memory", 0),
                "avg_gpu_utilization": result.get("avg_gpu_utilization", 0),
                "avg_gpu_memory_utilization": result.get("avg_gpu_memory_utilization", 0),
                "total_gpu_memory": result.get("total_gpu_memory", 0)
            }
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def create_visualizations(self):
        """Create visualizations of runtime scalability results."""
        print("Creating visualizations...")
        
        df_results = self.create_results_dataframe()
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Runtime Scalability Analysis\n{self.timestamp}', fontsize=16, fontweight='bold')
        
        # 1. Training time heatmap (markers vs samples, averaged over nodes)
        ax1 = axes[0, 0]
        pivot_time = df_results.groupby(['num_samples', 'num_markers'])['total_training_time'].mean().unstack()
        sns.heatmap(pivot_time, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Total Training Time (seconds)\n(Averaged over node counts)')
        ax1.set_xlabel('Number of Markers')
        ax1.set_ylabel('Number of Samples')
        
        # 2. Memory usage heatmap (markers vs samples, averaged over nodes)
        ax2 = axes[0, 1]
        pivot_memory = df_results.groupby(['num_samples', 'num_markers'])['peak_memory_usage'].mean().unstack()
        sns.heatmap(pivot_memory, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
        ax2.set_title('Peak Memory Usage (MB)\n(Averaged over node counts)')
        ax2.set_xlabel('Number of Markers')
        ax2.set_ylabel('Number of Samples')
        
        # 3. GPU memory heatmap (if available)
        ax3 = axes[0, 2]
        if 'peak_gpu_memory' in df_results.columns and df_results['peak_gpu_memory'].sum() > 0:
            pivot_gpu = df_results.groupby(['num_samples', 'num_markers'])['peak_gpu_memory'].mean().unstack()
            sns.heatmap(pivot_gpu, annot=True, fmt='.1f', cmap='Greens', ax=ax3)
            ax3.set_title('Peak GPU Memory (MB)\n(Averaged over node counts)')
        else:
            ax3.text(0.5, 0.5, 'GPU Memory\nNot Available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Peak GPU Memory (MB)')
        ax3.set_xlabel('Number of Markers')
        ax3.set_ylabel('Number of Samples')
        
        # 4. Training time scaling (avg nodes vs time, averaged over samples and markers)
        ax4 = axes[1, 0]
        node_time_avg = df_results.groupby('avg_nodes_per_sample')['total_training_time'].mean()
        ax4.plot(node_time_avg.index, node_time_avg.values, 
                marker='o', linewidth=2, markersize=6, color='red')
        ax4.set_xlabel('Average Number of Nodes per Sample')
        ax4.set_ylabel('Total Training Time (seconds)')
        ax4.set_title('Training Time vs Average Node Count\n(Averaged over samples and markers)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        # 5. Memory usage scaling (avg nodes vs memory, averaged over samples and markers)
        ax5 = axes[1, 1]
        node_memory_avg = df_results.groupby('avg_nodes_per_sample')['peak_memory_usage'].mean()
        ax5.plot(node_memory_avg.index, node_memory_avg.values, 
                marker='s', linewidth=2, markersize=6, color='blue')
        ax5.set_xlabel('Average Number of Nodes per Sample')
        ax5.set_ylabel('Peak Memory Usage (MB)')
        ax5.set_title('Memory Usage vs Average Node Count\n(Averaged over samples and markers)')
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        
        # 6. GPU utilization (if available)
        ax6 = axes[1, 2]
        if 'avg_gpu_utilization' in df_results.columns and df_results['avg_gpu_utilization'].sum() > 0:
            node_gpu_avg = df_results.groupby('avg_nodes_per_sample')['avg_gpu_utilization'].mean()
            ax6.plot(node_gpu_avg.index, node_gpu_avg.values, 
                    marker='^', linewidth=2, markersize=6, color='green')
            ax6.set_xlabel('Average Number of Nodes per Sample')
            ax6.set_ylabel('Average GPU Utilization (%)')
            ax6.set_title('GPU Utilization vs Average Node Count\n(Averaged over samples and markers)')
            ax6.grid(True, alpha=0.3)
            ax6.set_xscale('log')
        else:
            ax6.text(0.5, 0.5, 'GPU Utilization\nNot Available', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('GPU Utilization vs Average Node Count')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'runtime_scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create runtime summary table
        self._create_runtime_summary_table(df_results)
        
        print(f"Visualizations saved to {self.plots_dir}")
    
    def _create_runtime_summary_table(self, df_results):
        """Create a runtime summary table."""
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for _, row in df_results.iterrows():
            gpu_mem = f"{row.get('peak_gpu_memory', 0):.1f}MB" if 'peak_gpu_memory' in row else "N/A"
            gpu_util = f"{row.get('avg_gpu_utilization', 0):.1f}%" if row.get('avg_gpu_utilization', 0) > 0 else "N/A"
            
            summary_data.append([
                f"{row['num_markers']}",
                f"{row['num_samples']}",
                f"{row['avg_nodes_per_sample']:.0f}",
                f"{row['total_training_time']:.1f}s",
                f"{row['avg_epoch_time']:.2f}s",
                f"{row['peak_memory_usage']:.1f}MB",
                gpu_mem,
                gpu_util
            ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Markers', 'Samples', 'Avg Nodes', 'Total Time (s)', 'Avg Epoch (s)', 'Peak Memory (MB)', 'GPU Memory (MB)', 'GPU Util (%)'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        plt.title('Runtime Summary Table', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.plots_dir / 'runtime_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the scalability simulation."""
    print("GNN Runtime Scalability Simulation")
    print("=" * 50)
    
    # Create and run simulator
    simulator = ScalabilitySimulator()
    simulator.run_simulation()
    
    print(f"\nSimulation completed successfully!")
    print(f"Results saved to: {simulator.output_dir}")
    print(f"Check the plots directory for visualizations: {simulator.plots_dir}")


if __name__ == "__main__":
    main()
