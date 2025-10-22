#!/usr/bin/env python3
"""
Generate comprehensive summary report for GNN scalability simulation results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load simulation results from JSON file."""
    with open('/home/rifaioglu/projects/GNNClinicalOutcomePrediction/results/scalability_simulation_20250916_160503/data/simulation_results.json', 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def create_runtime_scalability_plot(df, ax):
    """Create runtime scalability analysis plot."""
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values='avg_epoch_time', 
        index='num_markers', 
        columns='num_samples', 
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Average Epoch Time (seconds)'})
    ax.set_title('Runtime Scalability Analysis\n(Average Epoch Time by Markers and Samples)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Number of Markers', fontsize=12)
    
    # Add performance annotations
    min_time = pivot_data.min().min()
    max_time = pivot_data.max().max()
    ax.text(0.02, 0.98, f'Fastest: {min_time:.2f}s\nSlowest: {max_time:.2f}s', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_memory_usage_plot(df, ax):
    """Create memory usage analysis plot."""
    # Create pivot table for memory usage
    pivot_data = df.pivot_table(
        values='peak_memory_usage', 
        index='num_markers', 
        columns='num_samples', 
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='Blues', ax=ax, cbar_kws={'label': 'Peak Memory Usage (MB)'})
    ax.set_title('Memory Usage Analysis\n(Peak Memory Usage by Markers and Samples)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Number of Markers', fontsize=12)
    
    # Add memory efficiency annotations
    min_memory = pivot_data.min().min()
    max_memory = pivot_data.max().max()
    ax.text(0.02, 0.98, f'Min: {min_memory:.1f}MB\nMax: {max_memory:.1f}MB', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_performance_trends_plot(df, ax):
    """Create performance trends plot."""
    # Plot average epoch time vs number of samples for different marker counts
    marker_counts = sorted(df['num_markers'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(marker_counts)))
    
    for i, markers in enumerate(marker_counts):
        subset = df[df['num_markers'] == markers].sort_values('num_samples')
        ax.plot(subset['num_samples'], subset['avg_epoch_time'], 
                marker='o', linewidth=2, markersize=6, 
                label=f'{markers} markers', color=colors[i])
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Average Epoch Time (seconds)', fontsize=12)
    ax.set_title('Performance Trends: Epoch Time vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

def create_memory_trends_plot(df, ax):
    """Create memory usage trends plot."""
    # Plot peak memory usage vs number of samples for different marker counts
    marker_counts = sorted(df['num_markers'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(marker_counts)))
    
    for i, markers in enumerate(marker_counts):
        subset = df[df['num_markers'] == markers].sort_values('num_samples')
        ax.plot(subset['num_samples'], subset['peak_memory_usage'], 
                marker='s', linewidth=2, markersize=6, 
                label=f'{markers} markers', color=colors[i])
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Usage Trends: Peak Memory vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

def create_efficiency_analysis_plot(df, ax):
    """Create computational efficiency analysis."""
    # Calculate efficiency metrics
    df['throughput'] = df['num_samples'] / df['avg_epoch_time']  # samples per second
    df['memory_efficiency'] = df['num_samples'] / df['peak_memory_usage']  # samples per MB
    
    # Create scatter plot
    scatter = ax.scatter(df['num_markers'], df['num_samples'], 
                       c=df['throughput'], s=df['peak_memory_usage']*2, 
                       alpha=0.7, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Markers', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Computational Efficiency Analysis\n(Size ∝ Memory Usage, Color ∝ Throughput)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Throughput (samples/second)', fontsize=10)
    
    # Add efficiency annotations
    max_throughput = df['throughput'].max()
    min_throughput = df['throughput'].min()
    ax.text(0.02, 0.98, f'Max Throughput: {max_throughput:.1f} samples/s\nMin Throughput: {min_throughput:.1f} samples/s', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def create_gpu_utilization_plot(df, ax):
    """Create GPU utilization analysis."""
    # Create pivot table for GPU utilization
    pivot_data = df.pivot_table(
        values='avg_gpu_utilization', 
        index='num_markers', 
        columns='num_samples', 
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'GPU Utilization (%)'}, vmin=80, vmax=100)
    ax.set_title('GPU Utilization Analysis\n(Average GPU Utilization by Configuration)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Number of Markers', fontsize=12)

def create_summary_statistics_table(df, ax):
    """Create summary statistics table."""
    ax.axis('off')
    
    # Calculate summary statistics
    stats_data = {
        'Metric': [
            'Total Configurations Tested',
            'Marker Range',
            'Sample Range', 
            'Fastest Epoch Time',
            'Slowest Epoch Time',
            'Min Memory Usage',
            'Max Memory Usage',
            'Average GPU Utilization',
            'Total Training Time (all configs)',
            'Average Nodes per Sample'
        ],
        'Value': [
            f"{len(df)}",
            f"{df['num_markers'].min()}-{df['num_markers'].max()}",
            f"{df['num_samples'].min()}-{df['num_samples'].max()}",
            f"{df['avg_epoch_time'].min():.2f}s",
            f"{df['avg_epoch_time'].max():.2f}s",
            f"{df['peak_memory_usage'].min():.1f}MB",
            f"{df['peak_memory_usage'].max():.1f}MB",
            f"{df['avg_gpu_utilization'].mean():.1f}%",
            f"{df['total_training_time'].sum()/3600:.1f}h",
            f"{df['avg_nodes_per_sample'].mean():.0f}"
        ]
    }
    
    # Create table
    table_data = list(zip(stats_data['Metric'], stats_data['Value']))
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                    cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax.set_title('Simulation Summary Statistics', fontsize=16, fontweight='bold', pad=20)

def generate_pdf_report(df):
    """Generate comprehensive PDF report."""
    output_path = '/home/rifaioglu/projects/GNNClinicalOutcomePrediction/results/scalability_simulation_20250916_160503/plots/simulation_summary_report.pdf'
    
    with PdfPages(output_path) as pdf:
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Summary Statistics Table
        ax1 = plt.subplot(3, 3, 1)
        create_summary_statistics_table(df, ax1)
        
        # 2. Runtime Scalability Heatmap
        ax2 = plt.subplot(3, 3, 2)
        create_runtime_scalability_plot(df, ax2)
        
        # 3. Memory Usage Heatmap
        ax3 = plt.subplot(3, 3, 3)
        create_memory_usage_plot(df, ax3)
        
        # 4. Performance Trends
        ax4 = plt.subplot(3, 3, 4)
        create_performance_trends_plot(df, ax4)
        
        # 5. Memory Trends
        ax5 = plt.subplot(3, 3, 5)
        create_memory_trends_plot(df, ax5)
        
        # 6. Efficiency Analysis
        ax6 = plt.subplot(3, 3, 6)
        create_efficiency_analysis_plot(df, ax6)
        
        # 7. GPU Utilization
        ax7 = plt.subplot(3, 3, 7)
        create_gpu_utilization_plot(df, ax7)
        
        # 8. Additional Analysis - Training Time vs Configuration
        ax8 = plt.subplot(3, 3, 8)
        config_labels = [f"{row['num_markers']}M-{row['num_samples']}S" for _, row in df.iterrows()]
        ax8.bar(range(len(df)), df['total_training_time'], color='skyblue', alpha=0.7)
        ax8.set_xlabel('Configuration (Markers-Samples)', fontsize=10)
        ax8.set_ylabel('Total Training Time (seconds)', fontsize=10)
        ax8.set_title('Total Training Time by Configuration', fontsize=12, fontweight='bold')
        ax8.tick_params(axis='x', rotation=45, labelsize=8)
        ax8.set_xticks(range(len(df)))
        ax8.set_xticklabels(config_labels, rotation=45, ha='right')
        
        # 9. GPU Memory Utilization
        ax9 = plt.subplot(3, 3, 9)
        gpu_mem_pivot = df.pivot_table(
            values='avg_gpu_memory_utilization', 
            index='num_markers', 
            columns='num_samples', 
            aggfunc='mean'
        )
        sns.heatmap(gpu_mem_pivot, annot=True, fmt='.3f', cmap='Purples', ax=ax9, 
                   cbar_kws={'label': 'GPU Memory Utilization'})
        ax9.set_title('GPU Memory Utilization Analysis', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Number of Samples', fontsize=10)
        ax9.set_ylabel('Number of Markers', fontsize=10)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create a second page with detailed analysis
        fig2 = plt.figure(figsize=(20, 16))
        
        # Performance vs Complexity Analysis
        ax1 = plt.subplot(2, 3, 1)
        df['complexity'] = df['num_markers'] * df['num_samples']
        ax1.scatter(df['complexity'], df['avg_epoch_time'], 
                   c=df['num_markers'], s=100, alpha=0.7, cmap='viridis')
        ax1.set_xlabel('Computational Complexity (Markers × Samples)')
        ax1.set_ylabel('Average Epoch Time (seconds)')
        ax1.set_title('Performance vs Computational Complexity')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
        cbar1.set_label('Number of Markers')
        
        # Memory vs Complexity Analysis
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(df['complexity'], df['peak_memory_usage'], 
                   c=df['num_markers'], s=100, alpha=0.7, cmap='plasma')
        ax2.set_xlabel('Computational Complexity (Markers × Samples)')
        ax2.set_ylabel('Peak Memory Usage (MB)')
        ax2.set_title('Memory Usage vs Computational Complexity')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('Number of Markers')
        
        # Efficiency Analysis
        ax3 = plt.subplot(2, 3, 3)
        df['efficiency'] = df['num_samples'] / (df['avg_epoch_time'] * df['peak_memory_usage'])
        ax3.scatter(df['num_markers'], df['efficiency'], 
                   c=df['num_samples'], s=100, alpha=0.7, cmap='coolwarm')
        ax3.set_xlabel('Number of Markers')
        ax3.set_ylabel('Computational Efficiency')
        ax3.set_title('Computational Efficiency by Marker Count')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        cbar3 = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar3.set_label('Number of Samples')
        
        # GPU Utilization Trends
        ax4 = plt.subplot(2, 3, 4)
        for markers in sorted(df['num_markers'].unique()):
            subset = df[df['num_markers'] == markers].sort_values('num_samples')
            ax4.plot(subset['num_samples'], subset['avg_gpu_utilization'], 
                    marker='o', label=f'{markers} markers', linewidth=2)
        ax4.set_xlabel('Number of Samples')
        ax4.set_ylabel('GPU Utilization (%)')
        ax4.set_title('GPU Utilization Trends')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Memory Efficiency Analysis
        ax5 = plt.subplot(2, 3, 5)
        df['memory_efficiency'] = df['num_samples'] / df['peak_memory_usage']
        ax5.scatter(df['num_markers'], df['memory_efficiency'], 
                   c=df['num_samples'], s=100, alpha=0.7, cmap='RdYlBu')
        ax5.set_xlabel('Number of Markers')
        ax5.set_ylabel('Memory Efficiency (samples/MB)')
        ax5.set_title('Memory Efficiency Analysis')
        ax5.set_xscale('log')
        cbar5 = plt.colorbar(ax5.collections[0], ax=ax5)
        cbar5.set_label('Number of Samples')
        
        # Performance Scaling Analysis
        ax6 = plt.subplot(2, 3, 6)
        # Calculate scaling factors
        baseline = df[(df['num_markers'] == 30) & (df['num_samples'] == 50)].iloc[0]
        df['time_scaling'] = df['avg_epoch_time'] / baseline['avg_epoch_time']
        df['memory_scaling'] = df['peak_memory_usage'] / baseline['peak_memory_usage']
        
        ax6.scatter(df['time_scaling'], df['memory_scaling'], 
                   c=df['num_markers'], s=100, alpha=0.7, cmap='Set1')
        ax6.set_xlabel('Time Scaling Factor')
        ax6.set_ylabel('Memory Scaling Factor')
        ax6.set_title('Performance Scaling Analysis')
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        cbar6 = plt.colorbar(ax6.collections[0], ax=ax6)
        cbar6.set_label('Number of Markers')
        
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches='tight', dpi=300)
        plt.close()
    
    return output_path

def main():
    """Main function to generate the report."""
    print("Loading simulation data...")
    df = load_data()
    
    print("Generating comprehensive PDF report...")
    output_path = generate_pdf_report(df)
    
    print(f"Report generated successfully: {output_path}")
    
    # Print summary statistics
    print("\n=== SIMULATION SUMMARY ===")
    print(f"Total configurations tested: {len(df)}")
    print(f"Marker range: {df['num_markers'].min()}-{df['num_markers'].max()}")
    print(f"Sample range: {df['num_samples'].min()}-{df['num_samples'].max()}")
    print(f"Fastest epoch time: {df['avg_epoch_time'].min():.2f}s")
    print(f"Slowest epoch time: {df['avg_epoch_time'].max():.2f}s")
    print(f"Min memory usage: {df['peak_memory_usage'].min():.1f}MB")
    print(f"Max memory usage: {df['peak_memory_usage'].max():.1f}MB")
    print(f"Average GPU utilization: {df['avg_gpu_utilization'].mean():.1f}%")

if __name__ == "__main__":
    main()





