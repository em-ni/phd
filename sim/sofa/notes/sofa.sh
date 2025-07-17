#!/bin/bash

# Initialize the shell to work with Conda
conda init bash

# Activate the 'sofa' Anaconda environment
conda activate sim

# Set the 'SOFA_ROOT' environment variable
export SOFA_ROOT=${HOME}/Desktop/github/sim/sofa/build/v23.12

# Set the 'PYTHONPATH' environment variable
export PYTHONPATH=${HOME}/Desktop/github/sim/sofa/build/v23.12/lib/python3/site-packages:$PYTHONPATH

# Print the values of the variables
echo "SOFA_ROOT: $SOFA_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Environment initialized" 