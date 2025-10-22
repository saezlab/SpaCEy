#!/usr/bin/env python3
"""
Test script for the scalability simulation.

This script runs a minimal version of the scalability simulation to verify
that all components work correctly.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scalability_simulation import ScalabilitySimulator


def create_test_config():
    """Create a minimal test configuration."""
    test_config = {
        "marker_counts": [5, 10],
        "sample_counts": [10, 20],
        "epochs_per_config": 2,
        "model_config": {
            "model_type": "GATV2",
            "num_gcn_layers": 1,
            "num_ff_layers": 1,
            "gcn_hidden_neurons": 16,
            "ff_hidden_neurons": 32,
            "dropout": 0.1,
            "heads": 2
        },
        "training_config": {
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "loss_function": "MSE"
        },
        "synthetic_data_config": {
            "num_nodes_per_sample": 20,
            "edge_probability": 0.2,
            "feature_noise": 0.1,
            "label_noise": 0.05
        },
        "random_seed": 42
    }
    
    return test_config


def test_simulation():
    """Run a minimal test of the scalability simulation."""
    print("Testing scalability simulation...")
    
    # Create test configuration
    test_config = create_test_config()
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, indent=2)
        config_path = f.name
    
    try:
        # Create simulator with test config
        simulator = ScalabilitySimulator(config_path)
        
        # Override output directory to avoid cluttering results
        simulator.output_dir = Path("test_output")
        simulator.plots_dir = simulator.output_dir / "plots"
        simulator.data_dir = simulator.output_dir / "data"
        
        # Create directories
        simulator.output_dir.mkdir(exist_ok=True)
        simulator.plots_dir.mkdir(exist_ok=True)
        simulator.data_dir.mkdir(exist_ok=True)
        
        print(f"Test output directory: {simulator.output_dir}")
        
        # Run simulation
        simulator.run_simulation()
        
        # Check that results were generated
        results_file = simulator.data_dir / "simulation_results.json"
        csv_file = simulator.data_dir / "simulation_results.csv"
        
        if results_file.exists() and csv_file.exists():
            print("✓ Results files generated successfully")
            
            # Load and check results
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"✓ Generated {len(results)} configuration results")
            
            # Check that we have the expected number of configurations
            expected_configs = len(test_config["marker_counts"]) * len(test_config["sample_counts"])
            if len(results) == expected_configs:
                print(f"✓ Correct number of configurations tested ({expected_configs})")
            else:
                print(f"✗ Expected {expected_configs} configurations, got {len(results)}")
                return False
            
            # Check that plots were generated
            plot_files = list(simulator.plots_dir.glob("*.png"))
            if plot_files:
                print(f"✓ Generated {len(plot_files)} plot files")
            else:
                print("✗ No plot files generated")
                return False
            
            print("✓ All tests passed!")
            return True
            
        else:
            print("✗ Results files not generated")
            return False
    
    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary config file
        if os.path.exists(config_path):
            os.unlink(config_path)


if __name__ == "__main__":
    success = test_simulation()
    if success:
        print("\n🎉 Test completed successfully!")
        print("The scalability simulation is working correctly.")
    else:
        print("\n❌ Test failed!")
        print("Please check the error messages above.")
        sys.exit(1)
