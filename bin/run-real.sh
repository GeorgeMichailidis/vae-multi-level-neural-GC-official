#!/bin/bash

ds_str=${1-EEG_EC}
cuda=${2-0}

run_MultiSubVAE=true
run_OneSubVAE=true
version=''

cd ..
pwd; hostname;
echo "Running VAE-based methods for ds_str=$ds_str using cuda=$cuda"

if [[ "$ds_str" == 'EEG_EC' ]]; then
    n=({0..21})
elif [[ "$ds_str" == 'EEG_EO' ]]; then
    n=({0..20})
else
    echo "unsupported ds_str"
    exit 1
fi

for train_size in 30000
do
    if $run_MultiSubVAE ; then
        config="configs/${ds_str}${version}.yaml"
        output_dir="output_real/${ds_str}-MultiSubVAE-${train_size}${version}"
        echo "####################"
        date
        CMD="python -u train.py --ds_str=$ds_str --cuda=$cuda --train_size=$train_size --config=$config --output_dir=$output_dir"
        echo "CMD: ${CMD}"
        echo "####################"
        eval ${CMD}
    fi
    
    ## run one-subject
    if $run_OneSubVAE ; then
        config="configs/${ds_str}_oneSub${version}.yaml"
        for subject_id in "${n[@]}"; do
            output_dir="output_real/${ds_str}-OneSubVAE-${train_size}${version}/subject_${subject_id}"
            echo "####################"
            date
            CMD="python -u train_one.py --ds_str=$ds_str --cuda=$cuda --subject_id=$subject_id --train_size=$train_size --config=$config --output_dir=$output_dir"
            echo "CMD: ${CMD}"
            echo "####################"
            eval ${CMD}
        done
    fi
    
done

date
