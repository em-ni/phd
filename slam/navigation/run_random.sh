#!/bin/bash
# Usage: ./run_random_autopilot.sh N
# Runs the simulation N times with random selection of 2 centerlines in autopilot mode

if [ -z "$1" ]; then
  echo "Usage: $0 N"
  exit 1
fi

N=$1
for ((i=1; i<=N; i++)); do
  echo "Run $i/$N:"
  python main.py -view fp -random 2 -autopilot -record True
  echo "---"
done
