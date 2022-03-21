#!/bin/bash

#SBATCH --job-name="w-run"
#SBATCH --output="out/run.%A.%N.out"
#SBATCH --error="out/run.%A.%N.err"
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH -t 72:00:00

source ~/window/window-env/bin/activate

python run.py $config
