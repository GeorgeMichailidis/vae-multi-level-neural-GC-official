#!/bin/bash
trap "EXIT" INT

ds=${1-LinearVAR}
cuda=${2-0}
version=${3}

generate_data=true
run_simMultiSubVAE=true
run_simOneSubVAE=true

if [[ "$ds" == "LinearVAR" ]]; then
    ds_strs=('LinearVAR1' 'LinearVAR2' 'LinearVAR3' 'LinearVAR4' 'LinearVAR5')
    train_sizes=(200 1000)
    n=({0..19})
elif [[ "$ds" == "NonLinearVAR" ]]; then
    ds_strs=('NonLinearVAR1' 'NonLinearVAR2' 'NonLinearVAR3' 'NonLinearVAR4' 'NonLinearVAR5')
    train_sizes=(500 2000)
    n=({0..9})
else
    echo "unsupported ds"
    exit 1
fi

echo "####################"
date
echo "running additional experiments for varying degree of heterogeneity"
echo "ds=${ds}; generate_data=${generate_data}"
echo "####################"

data_seeds=(0)

cd ..
pwd; hostname;

for ds_str in "${ds_strs[@]}";
do
    config="configs/${ds}_extra/${ds_str}${version}.yaml"
    for seed in "${data_seeds[@]}";
    do
        ########## generate data ################
        if $generate_data; then
            python -u generator/generate_VAR.py --ds_str=$ds_str --config=$config --seed=$seed
        fi
        
        for train_size in "${train_sizes[@]}";
        do
            
            ## joint learning on synthetic data
            if $run_simMultiSubVAE ; then
                config="configs/${ds}_extra/${ds_str}${version}.yaml"
                output_dir="output_sim/${ds}_extra/${ds_str}_seed${seed}-simMultiSubVAE-${train_size}${version}"
                echo "####################"
                date
                CMD="python -u run_sim.py --ds_str=$ds_str --cuda=$cuda --train_size=$train_size --seed=$seed --config=$config --output_dir=$output_dir --eval-test-only"
                echo "CMD: ${CMD}"
                echo "####################"
                eval ${CMD} || { echo 'some error occurred; exit 1' ; exit 1; }
            fi
            
            ## individual learning
            if $run_simOneSubVAE ; then
                config="configs/${ds}_extra/${ds_str}_oneSub${version}.yaml"
                for subject_id in "${n[@]}";
                do
                    output_dir="output_sim/${ds}_extra/${ds_str}_seed${seed}-simOneSubVAE-${train_size}${version}/subject_${subject_id}"
                    echo "####################"
                    date
                    CMD="python -u run_simOne.py --ds_str=$ds_str --cuda=$cuda --subject_id=$subject_id --train_size=$train_size --seed=$seed --config=$config --output_dir=$output_dir --eval-test-only"
                    echo "CMD: ${CMD}"
                    echo "####################"
                    eval ${CMD} || { echo 'some error occurred; exit 1' ; exit 1; }
                done
            fi
        
        done
    done
done

date
