#!/bin/bash

#SBATCH --job-name="window"
#SBATCH --output="out/submit.%A.%N.out"
#SBATCH --error="out/submit.%A.%N.err"
#SBATCH --ntasks=6
#SBATCH --export=ALL
#SBATCH -t 72:00:00

source ~/window/window-env/bin/activate

configs=("all_normals" "all_underhangs" "all_overhangs" "all_imbalance" "all_vertical_misalignments" "all_horizontal_misalignments")

for c in ${configs[@]}; do
	sbatch --export=config="/home/nprzybylski/window/config/${c}.yaml" submit.slurm
done
