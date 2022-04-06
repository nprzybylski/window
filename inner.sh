#!/bin/bash

#SBATCH --job-name="w-inner"
#SBATCH --output="out/run.%A.%N.out"
#SBATCH --error="out/run.%A.%N.err"
#SBATCH --partition=IvyBridge
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH -t 96:00:00

#echo "model: $model"
source window-env/bin/activate

python inner.py $i $p0 $p1 $model $config $util $wpath $path $fpath $n_iters
