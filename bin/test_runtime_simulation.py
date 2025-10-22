#!/usr/bin/env python3
"""
Test script for the runtime scalability simulation.

This script runs a minimal version of the runtime simulation to verify
that all components work correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scalability_simulation import ScalabilitySimulator


def test_runtime_simulation():
    """Run a minimal test of the runtime simulation."""
    print("Testing runtime scalability simulation...")
    
    try:
        # Create simulator
        simulator = ScalabilitySimulator()
        
        # Override configuration for quick testing
        simulator.marker_counts = [30, 50]
        simulator.sample_counts = [50, 100]
        simulator.node_distribution = {
            "mean": 300,   # Smaller mean for testing
            "std": 100,    # Smaller std for testing
            "min": 100,    # Minimum nodes
            "max": 500     # Maximum nodes
        }
        simulator.epochs_per_config = 2
        
        # Override output directory to avoid cluttering results
        simulator.output_dir = Path("test_runtime_output")
        simulator.plots_dir = simulator.output_dir / "plots"
        simulator.data_dir = simulator.output_dir / "data"
        
        # Create directories
        simulator.output_dir.mkdir(exist_ok=True)
        simulator.plots_dir.mkdir(exist_ok=True)
        simulator.data_dir.mkdir(exist_ok=True)
        
        print(f"Test output directory: {simulator.output_dir}")
        print(f"Testing {len(simulator.marker_counts)} marker counts: {simulator.marker_counts}")
        print(f"Testing {len(simulator.sample_counts)} sample counts: {simulator.sample_counts}")
        print(f"Testing node counts with normal distribution: mean={simulator.node_distribution['mean']}, std={simulator.node_distribution['std']}")
        
        # Run simulation
        simulator.run_simulation()
        
        # Check that results were generated
        results_file = simulator.data_dir / "simulation_results.json"
        csv_file = simulator.data_dir / "simulation_results.csv"
        
        if results_file.exists() and csv_file.exists():
            print("✓ Results files generated successfully")
            
            # Load and check results
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"✓ Generated {len(results)} configuration results")
            
            # Check that we have the expected number of configurations
            expected_configs = len(simulator.marker_counts) * len(simulator.sample_counts)
            if len(results) == expected_configs:
                print(f"✓ Correct number of configurations tested ({expected_configs})")
            else:
                print(f"✗ Expected {expected_configs} configurations, got {len(results)}")
                return False
            
            # Check that plots were generated
            plot_files = list(simulator.plots_dir.glob("*.png"))
            if plot_files:
                print(f"✓ Generated {len(plot_files)} plot files")
                for plot_file in plot_files:
                    print(f"  - {plot_file.name}")
            else:
                print("✗ No plot files generated")
                return False
            
            # Check runtime data
            for result in results:
                if 'total_training_time' in result and 'peak_memory_usage' in result:
                    print(f"✓ Configuration {result['config_name']}: "
                          f"Time={result['total_training_time']:.1f}s, "
                          f"Memory={result['peak_memory_usage']:.1f}MB")
                else:
                    print(f"✗ Missing runtime data in {result['config_name']}")
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


if __name__ == "__main__":
    success = test_runtime_simulation()
    if success:
        print("\n🎉 Runtime simulation test completed successfully!")
        print("The runtime scalability simulation is working correctly.")
    else:
        print("\n❌ Runtime simulation test failed!")
        print("Please check the error messages above.")
        sys.exit(1)
