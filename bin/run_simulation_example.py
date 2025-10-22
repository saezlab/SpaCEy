#!/usr/bin/env python3
"""
Example script demonstrating how to run the scalability simulation.

This script shows different ways to run the simulation with various configurations.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scalability_simulation import ScalabilitySimulator


def example_1_quick_test():
    """Example 1: Run a quick test with minimal configurations."""
    print("=" * 60)
    print("Example 1: Quick Test")
    print("=" * 60)
    
    # Create a minimal configuration for quick testing
    quick_config = {
        "marker_counts": [10, 50],
        "sample_counts": [50, 100],
        "epochs_per_config": 3,
        "model_config": {
            "model_type": "GATV2",
            "num_gcn_layers": 2,
            "num_ff_layers": 2,
            "gcn_hidden_neurons": 32,
            "ff_hidden_neurons": 64,
            "dropout": 0.2,
            "heads": 2
        },
        "training_config": {
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 16,
            "loss_function": "MSE"
        },
        "synthetic_data_config": {
            "num_nodes_per_sample": 50,
            "edge_probability": 0.1,
            "feature_noise": 0.1,
            "label_noise": 0.05
        },
        "random_seed": 42
    }
    
    # Save configuration to file
    config_file = "example_quick_config.json"
    with open(config_file, 'w') as f:
        json.dump(quick_config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    print("This will test 4 configurations (2 markers × 2 samples) with 3 epochs each.")
    print("Expected runtime: ~2-5 minutes")
    
    # Run simulation
    simulator = ScalabilitySimulator(config_file)
    simulator.run_simulation()
    
    print(f"Results saved to: {simulator.output_dir}")
    print()


def example_2_comprehensive_test():
    """Example 2: Run a comprehensive test with more configurations."""
    print("=" * 60)
    print("Example 2: Comprehensive Test")
    print("=" * 60)
    
    # Create a comprehensive configuration
    comprehensive_config = {
        "marker_counts": [10, 25, 50, 100, 200],
        "sample_counts": [50, 100, 200, 500],
        "epochs_per_config": 10,
        "model_config": {
            "model_type": "GATV2",
            "num_gcn_layers": 2,
            "num_ff_layers": 2,
            "gcn_hidden_neurons": 64,
            "ff_hidden_neurons": 128,
            "dropout": 0.2,
            "heads": 4
        },
        "training_config": {
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "loss_function": "MSE"
        },
        "synthetic_data_config": {
            "num_nodes_per_sample": 100,
            "edge_probability": 0.1,
            "feature_noise": 0.1,
            "label_noise": 0.05
        },
        "random_seed": 42
    }
    
    # Save configuration to file
    config_file = "example_comprehensive_config.json"
    with open(config_file, 'w') as f:
        json.dump(comprehensive_config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    print("This will test 20 configurations (5 markers × 4 samples) with 10 epochs each.")
    print("Expected runtime: ~30-60 minutes")
    
    # Run simulation
    simulator = ScalabilitySimulator(config_file)
    simulator.run_simulation()
    
    print(f"Results saved to: {simulator.output_dir}")
    print()


def example_3_custom_analysis():
    """Example 3: Custom analysis focusing on specific aspects."""
    print("=" * 60)
    print("Example 3: Custom Analysis")
    print("=" * 60)
    
    # Create a configuration focused on memory usage analysis
    memory_config = {
        "marker_counts": [10, 20, 50, 100, 200, 500],
        "sample_counts": [25, 50, 100, 200],
        "epochs_per_config": 5,
        "model_config": {
            "model_type": "PNAConv",  # Different model type
            "num_gcn_layers": 3,
            "num_ff_layers": 2,
            "gcn_hidden_neurons": 128,
            "ff_hidden_neurons": 256,
            "dropout": 0.3,
            "heads": 1  # Not used for PNAConv
        },
        "training_config": {
            "learning_rate": 0.0005,
            "weight_decay": 1e-4,
            "batch_size": 16,  # Smaller batch size to test memory limits
            "loss_function": "MSE"
        },
        "synthetic_data_config": {
            "num_nodes_per_sample": 150,
            "edge_probability": 0.15,
            "feature_noise": 0.05,
            "label_noise": 0.02
        },
        "random_seed": 42
    }
    
    # Save configuration to file
    config_file = "example_memory_analysis_config.json"
    with open(config_file, 'w') as f:
        json.dump(memory_config, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    print("This configuration focuses on memory usage analysis with PNAConv model.")
    print("Tests 24 configurations with varying complexity.")
    print("Expected runtime: ~20-40 minutes")
    
    # Run simulation
    simulator = ScalabilitySimulator(config_file)
    simulator.run_simulation()
    
    print(f"Results saved to: {simulator.output_dir}")
    print()


def example_4_command_line_usage():
    """Example 4: Show command line usage examples."""
    print("=" * 60)
    print("Example 4: Command Line Usage")
    print("=" * 60)
    
    print("You can also run the simulation from the command line:")
    print()
    
    commands = [
        "# Run with default configuration",
        "python scalability_simulation.py",
        "",
        "# Run with custom configuration file",
        "python scalability_simulation.py --config simulation_config.json",
        "",
        "# Run quick test (fewer configurations)",
        "python scalability_simulation.py --quick",
        "",
        "# Run with specific configuration file",
        "python scalability_simulation.py --config example_quick_config.json"
    ]
    
    for cmd in commands:
        print(cmd)
    
    print()
    print("The --quick flag automatically reduces the number of configurations")
    print("and epochs for faster testing.")
    print()


def main():
    """Main function to run examples."""
    print("Scalability Simulation Examples")
    print("=" * 60)
    print()
    print("This script demonstrates different ways to run the scalability simulation.")
    print("Choose an example to run:")
    print()
    print("1. Quick Test (2-5 minutes)")
    print("2. Comprehensive Test (30-60 minutes)")
    print("3. Custom Memory Analysis (20-40 minutes)")
    print("4. Show Command Line Usage")
    print("5. Run All Examples")
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()
            
            if choice == "0":
                print("Exiting...")
                break
            elif choice == "1":
                example_1_quick_test()
            elif choice == "2":
                example_2_comprehensive_test()
            elif choice == "3":
                example_3_custom_analysis()
            elif choice == "4":
                example_4_command_line_usage()
            elif choice == "5":
                print("Running all examples...")
                example_1_quick_test()
                example_2_comprehensive_test()
                example_3_custom_analysis()
                example_4_command_line_usage()
            else:
                print("Invalid choice. Please enter a number between 0 and 5.")
                continue
            
            # Ask if user wants to continue
            if choice in ["1", "2", "3", "5"]:
                continue_choice = input("\nRun another example? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()
