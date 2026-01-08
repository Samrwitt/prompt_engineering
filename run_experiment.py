#!/usr/bin/env python3
"""
Quick Start Script for Prompt Engineering Experiment

This script provides an easy way to run the experiment with different configurations.
"""

import sys
from src.experiment import main

if __name__ == "__main__":
    print("=" * 70)
    print("Prompt Engineering Experiment - Quick Start")
    print("=" * 70)
    print("\nThis will run the full experiment across all datasets.")
    print("Expected runtime: 30-60 minutes on CPU\n")
    
    # Parse command line arguments
    run_block_analysis = "--no-block-analysis" not in sys.argv
    run_dspy = "--no-dspy" not in sys.argv
    save_curves = "--no-curves" not in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python run_experiment.py [options]")
        print("\nOptions:")
        print("  --no-block-analysis    Skip block effectiveness analysis")
        print("  --no-dspy              Skip DSPy SOTA baseline")
        print("  --no-curves            Don't save convergence curves")
        print("  -h, --help             Show this help message")
        sys.exit(0)
    
    try:
        main(
            run_block_analysis=run_block_analysis,
            run_dspy=run_dspy,
            save_curves=save_curves,
        )
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

