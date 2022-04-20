#!/bin/bash

source $wpath/window-env/bin/activate

configs=("all_normals" "all_underhangs" "all_overhangs" "all_imbalance" "all_vertical_misalignments" "all_horizontal_misalignments")

for c in ${configs[@]}; do
	sbatch --export=config="config/${c}_2500_250k_1sec.yaml",util=$util,wpath=$wpath,model=$model,vpath=$vpath,dpath=$dpath submit.slurm
done
