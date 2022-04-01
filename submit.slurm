#!/bin/bash

#SBATCH --job-name="w-run"
#SBATCH --output="out/run.%A.%N.out"
#SBATCH --error="out/run.%A.%N.err"
#SBATCH --partition=IvyBridge
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH -t 96:00:00

source /ac-project/nprzybylski/window/window-env/bin/activate

python run.py $config
