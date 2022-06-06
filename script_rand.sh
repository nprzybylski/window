#!/bin/bash

source $wpath/window-env/bin/activate

config="1sec_rand"

for ((i=0; i<50; i+=1))
do
	sbatch --export=config="config/${config}.yaml",util=$util,wpath=$wpath,model=$model,vpath=$vpath,dpath=$dpath,run=$i submit.slurm
done
