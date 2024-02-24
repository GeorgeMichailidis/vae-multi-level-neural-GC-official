#!/bin/bash

ds_str=${1-Lokta}
cuda=${2-0}
version=${3}

generate_data=true
run_simMultiSubVAE=true
run_simOneSubVAE=true

train_sizes=(10000)
data_seeds=(0 1 2 3 4)

cd ..
pwd; hostname;
echo "Running VAE-based methods for ds_str=$ds_str using cuda=$cuda"

for seed in "${data_seeds[@]}";
do
    ########## generate data ################
    if [[ "$ds_str" == 'LinearVAR' ]]; then
        n=({0..19})
        if $generate_data; then
            python -u generator/generate_VAR.py --ds_str=$ds_str --seed=$seed
        fi
    elif [[ "$ds_str" == 'Lorenz96' ]]; then
        n=({0..4})
        if $generate_data; then
            python -u generator/generate_Lorenz.py --ds_str=$ds_str --seed=$seed
        fi
    elif [[ "$ds_str" == 'NonLinearVAR' ]]; then
        n=({0..9})
        if $generate_data; then
            python -u generator/generate_VAR.py --ds_str=$ds_str --seed=$seed
        fi
    elif [[ "$ds_str" == 'Springs5' ]]; then
        n=({0..9})
        if $generate_data; then
            python -u generator/generate_springs.py --ds_str=$ds_str --seed=$seed --mp_cores=10
        fi
    elif [[ "$ds_str" == 'Lotka' ]]; then
        n=({0..9})
        if $generate_data; then
            python -u generator/generate_Lokta.py --ds_str=$ds_str --seed=$seed
        fi
    else
        echo "unsupported ds_str"
        exit 1
    fi
    
    ###########################
    
    for train_size in "${train_sizes[@]}";
    do
        ## run simMultiSubVAE
        if $run_simMultiSubVAE ; then
            config="configs/${ds_str}${version}.yaml"
            output_dir="output_sim/${ds_str}_seed${seed}-simMultiSubVAE-${train_size}${version}"
            echo "####################"
            date
            CMD="python -u run_sim.py --ds_str=$ds_str --cuda=$cuda --train_size=$train_size --seed=$seed --config=$config --output_dir=$output_dir --eval-test-only"
            echo "CMD: ${CMD}"
            echo "####################"
            eval ${CMD}
        fi
        
        ## run one-subject
        if $run_simOneSubVAE ; then
            config="configs/${ds_str}_oneSub${version}.yaml"
            for subject_id in "${n[@]}"; do
                output_dir="output_sim/${ds_str}_seed${seed}-simOneSubVAE-${train_size}${version}/subject_${subject_id}"
                echo "####################"
                date
                CMD="python -u run_simOne.py --ds_str=$ds_str --cuda=$cuda --subject_id=$subject_id --train_size=$train_size --seed=$seed --config=$config --output_dir=$output_dir --eval-test-only"
                echo "CMD: ${CMD}"
                echo "####################"
                eval ${CMD}
            done
        fi
        
    done
done

date
