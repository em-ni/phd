#!/bin/bash

# Usage: ./run_dataset.sh <config_directory> <number_of_runs>
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <config_directory> <number_of_runs>"
    exit 1
fi

CONFIG_DIR="$1"
NUM_RUNS="$2"
EXECUTABLE="./build/broncho_rgbd"  # Replace with the path to your executable if necessary

# Loop over each config file in the directory
for config in "$CONFIG_DIR"/*.ini; do
    echo " "
    echo " "
    echo "------------------------------------------------"
    echo "PROCESSING CONFIG FILE: $config"
    # Run the executable for the specified number of times
    for (( run=1; run<=NUM_RUNS; run++ )); do
        echo "-----------------------------------"
        echo "Run #$run for config: $config"
        $EXECUTABLE "$config"
        echo "-----------------------------------"
    done
    echo "------------------------------------------------"
done
echo "Use em/prepare_logs.py to prepare the logs for analysis with rpg_trajectory_evaluation"
echo "Done"
