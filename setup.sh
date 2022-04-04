#!/bin/bash

source /ac-project/nprzybylski/window/window-env/bin/activate
yq=/ac-project/nprzybylski/yq_linux_386

idxs=($($yq .idxs[] $wpath/$util))
S=($($yq .signals[] $wpath/$util))
columns=($($yq .columns[] $wpath/$util))
classDict=$($yq .classes $wpath/$util)

default=$($yq .default $wpath/$config)
plot_path=$($yq .default.plot_path $wpath/$config)
experiment_name=$($yq .default.experiment_name $wpath/$config)
path=$plot_path$experiment_name
model=$wpath/models/rfc1.joblib

if [ -d $path ]
then
    # echo "> Directory $path exists."
    echo ""
else
    # echo "> Directory $path does not exist. Creating now."
    mkdir $path
fi

sweep=$($yq .sweep $wpath/$config)
random_pick=$($yq .sweep.random_pick $wpath/$config)

# echo "> $random_pick"

file_idxs=()
if [ $random_pick ]
# can currently pick the same index more than once. Need to fix so there is no replacement
then
    n_files=$($yq .sweep.n_files $wpath/$config)
    for ((i=0; i<=$n_files; i++))
    do
        len_idx=${#idxs[@]}
        rand_idx=($(( RANDOM % len_idx )))
        file_idxs+=(${idxs[$rand_idx]})
    done
else
    file_idxs=($($yq .sweep.file_idxs[] $wpath/$config))
    n_files=${#file_idxs[@]}
fi

# echo $n_files
# echo ${file_idxs[@]}

width_per_step_lo=$($yq .sweep.width_per_step.lo $wpath/$config)
width_per_step_hi=$($yq .sweep.width_per_step.hi $wpath/$config)
width_per_step_step=$($yq .sweep.width_per_step.step $wpath/$config)

wps=()
for ((i=$width_per_step_lo; i<=$width_per_step_hi; i+=$width_per_step_step))
do
    # echo $i
    wps+=($i)
done
# echo ${wps[@]}

window_width_lo=$($yq .sweep.window_width.lo $wpath/$config)
window_width_hi=$($yq .sweep.window_width.hi $wpath/$config)
window_width_step=$($yq .sweep.window_width.step $wpath/$config)

ww=()
for ((i=$window_width_lo; i<=$window_width_hi; i+=$window_width_step))
do
    # echo $i
    ww+=($i)
done
# echo ${ww[@]}

wps_len=${#wps[@]}
ww_len=${#ww[@]}

param=()
n_iters=$((wps_len*ww_len))
r=0
for ((i=0; i<=$wps_len; i++))
do

    for ((j=0; j<=$ww_len; j++))
    do
        # param=()
        # param+=(${wps[i]})
        # param+=(${ww[j]})
        # not really needed, can just run python script from here
        # param=(${wps[$i]} ${ww[$j]})

        #sbatch --export=r=$r,param=$param,file_idxs=$file_idxs,model=$model,columns=$columns,n_files=$n_files,classDict=$classDict,n_iters=$n_iters,sweep=$sweep,path=$path s.slurm 
        sbatch --export=r=$r,param=$param,model=$model,path=$path,file_idxs=$file_idxs,columns=$columns,n_files=$n_files,n_iters=$n_iters,sweep=$sweep s.slurm
	# echo $r

        r=$((r+1))

    done

done

echo ${params[@]}
