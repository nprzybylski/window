#!/bin/bash

source /ac-project/nprzybylski/window/window-env/bin/activate

configs=("all_normals" "all_underhangs" "all_overhangs" "all_imbalance" "all_vertical_misalignments" "all_horizontal_misalignments")

for c in ${configs[@]}; do
	sbatch --export=config="/ac-project/nprzybylski/window/config/${c}_2500_250k.yaml" submit.slurm
done
