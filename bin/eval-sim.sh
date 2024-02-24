#!/bin/bash

cd ..
pwd; hostname;

ds_str=${1-LinearVAR}

data_seeds=(0)
for seed in "${data_seeds[@]}";
do
    echo "####################"
    date
    
    CMD="python -u eval_sim.py --ds_str=$ds_str --seed=$seed"
    
    echo "CMD: ${CMD}"
    echo "####################"
    eval ${CMD}

done
