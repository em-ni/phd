#!/bin/bash

# INPUT_1: path to the input file
# INPUT_2: path to the output file

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 INPUT_1 INPUT_2"
    exit 1
fi

# Assign arguments to variables
INPUT_1=$1
INPUT_2=$2

# Define the path to the Conda setup script
CONDA_SETUP_SCRIPT="/home/emanuele/anaconda3/etc/profile.d/conda.sh"

# Open a new terminal and run the commands
gnome-terminal -- bash -c "source $CONDA_SETUP_SCRIPT; conda activate vmtk; vmtkcenterlines -ifile $INPUT_1 -ofile $INPUT_2; echo 'Centerline saved at $INPUT_2'; exec bash"

# Example
# ./centerline.sh data/mesh/vascularmodel/0023_H_AO_MFS/Models/0131_0000.vtp test.vtp
