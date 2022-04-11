#!/bin/bash

source $wpath/window-env/bin/activate

configs=("all_normals" "all_underhangs" "all_overhangs" "all_imbalance" "all_vertical_misalignments" "all_horizontal_misalignments")

name=$1

for c in ${configs[@]}; do
	sbatch --export=config="${wpath}/config/${c}_2500_250k.yaml",name=$name submit.slurm
done
