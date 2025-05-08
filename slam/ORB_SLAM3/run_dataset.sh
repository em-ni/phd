#!/bin/bash

# Usage: ./run_dataset.sh <dataset_folder> <number_of_runs>
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_folder> <number_of_runs>"
    exit 1
fi

DATASET_FOLDER="$1"
DATASET_NAME=$(basename "$DATASET_FOLDER")
echo "Dataset name: $DATASET_NAME"
echo "Number of runs: $2"
CONFIG_DIR="$DATASET_FOLDER/configs"
NUM_RUNS="$2"
EXECUTABLE="./build/broncho_rgbd"  

# Loop over each config file in the directory
for config in "$CONFIG_DIR"/*.ini; do
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
echo "Done"
echo " "
echo "Use em/prepare_logs.py to prepare the logs for analysis with rpg_trajectory_evaluation"
echo "e.g. python prepare_logs.py $DATASET_NAME rgbd-rd $DATASET_FOLDER"