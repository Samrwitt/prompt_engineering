#!/bin/bash
# run_experiment.sh

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Add project root to PYTHONPATH explicitly (optional given the sys.path fix, but good practice)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the experiment script
python src/experiment.py
