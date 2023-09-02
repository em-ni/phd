#!/bin/bash

# Initialize the shell to work with Conda
conda init bash

# Activate the 'sofa' Anaconda environment
conda activate sofa

# Set the 'SOFA_ROOT' environment variable
export SOFA_ROOT=/home/emanuele/Desktop/github/test/sofa/build/v22.12

# Set the 'PYTHONPATH' environment variable
export PYTHONPATH=/home/emanuele/Desktop/github/test/sofa/build/v22.12/lib/python3/site-packages:$PYTHONPATH

# Print the values of the variables
echo "SOFA_ROOT: $SOFA_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Environment initialized" 
