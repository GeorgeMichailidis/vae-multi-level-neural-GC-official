#!/usr/bin/env python3

"""
Read real data (EEG) and prepare samples in the same format as simulated data
python process_EEG.py
or
python process_EEG.py --ds_str='EEG_EC'
"""

import os
import sys
sys.path.append(os.getcwd())

import argparse
import yaml
import numpy as np
import pandas as pd
import shutil
import pickle
import datetime
import importlib
import glob
import re

from utils.utils_data import parse_trajectories, seq_to_samples

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ds_str', type=str, help='datasets to process, separated by comma',default='EEG_EC,EEG_EO')
parser.add_argument('--generate-summary-only', action='store_true', help='only generate summary statistics for raw data')
parser.add_argument('--verbose', action='store_true')

def main():

    global args
    args = parser.parse_args()

    print(f"python={'.'.join(map(str,sys.version_info[:3]))}")
    print(f"numpy={np.__version__}")
    print(f"pandas={pd.__version__}")

    parent_folder = 'data_real' ## just in case there are more data coming in
    raw_data_folder = os.path.join(parent_folder, 'EEG_CSV_files')

    datasets_to_process = args.ds_str.split(',')

    for ds in datasets_to_process:

        stimuli = ds.split('_')[-1]
        config = f'configs/{ds}.yaml'
        print(f'##### [{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] PROCESSING {ds}, config_file={config}')

        with open(config) as f:
            params = yaml.safe_load(f)['raw_data_specs']

        folder_name = os.path.join(parent_folder, ds)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f'folder={folder_name} created')

        ###########
        ## generate data summary
        ###########
        data_summary = generate_summary(raw_data_folder, stimuli)
        path_summary = os.path.join(folder_name, f'{stimuli}_data_summary.csv')
        data_summary.to_csv(path_summary,index=False)
        print(f'>> summary exported:{path_summary}; min_length={data_summary["T"].min()}, max_length={data_summary["T"].max()}')

        if args.generate_summary_only:
            continue

        ###########
        ## read and save down trajectories for each subject
        ###########
        assert params['num_nodes'] == len(params['nodes'])
        trajs, subject_id_map = read_raw_data(raw_data_folder, params['T_obs'], stimuli, params['nodes'])
        
        with open(os.path.join(folder_name, f'subject_id_map.pickle'), 'wb') as handle:
            pickle.dump(subject_id_map, handle, protocol = pickle.HIGHEST_PROTOCOL)
        ## save down trajectories
        with open(os.path.join(folder_name, f'trajs.pickle'), 'wb') as handle:
            pickle.dump(trajs, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        ##################
        ## save down also a copy of the sub-sampled trajectory
        ## note that the subsampling mechanism here ensures that the sub-sampled trajectories are completely non-overlapping
        trajs_subsampled = {}
        for subj_id, subj_traj in trajs.items():
            trajs_subsampled[subj_id] = create_subsample(subj_traj, sample_freq=params['sample_freq'])
        with open(os.path.join(folder_name, f'trajs_subsampled.pickle'), 'wb') as handle:
            pickle.dump(trajs_subsampled, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        ###########
        ## parse trajectories and prep samples
        ###########
        print(f'>> preparing sampels based on trajectories')

        for folder_type in ['train','val','test']:
            folder_name_typed = os.path.join(folder_name, folder_type)
            if not os.path.exists(folder_name_typed):
                os.mkdir(folder_name_typed)
                print(f'folder={folder_name_typed} created')
            else:
                print(f'folder={folder_name_typed} existed; emptied and recreated')
                shutil.rmtree(folder_name_typed)
                os.mkdir(folder_name_typed)

        num_val, num_test = params['num_val'], params['num_test']
        trajs_train, trajs_val, trajs_test = parse_trajectories(trajs,
                                                            num_val,
                                                            num_test,
                                                            context_length = params['context_length'],
                                                            prediction_length = params['target_length'],
                                                            stride = params['stride'],
                                                            sample_freq = params['sample_freq'],
                                                            verbose=args.verbose)

        ## save each sample as an .npy
        for sample_id in range(trajs_train.shape[0]):
            np.save(f'{folder_name}/train/data_{sample_id}.npy', trajs_train[sample_id])
        for sample_id in range(trajs_val.shape[0]):
            np.save(f'{folder_name}/val/data_{sample_id}.npy', trajs_val[sample_id])
        for sample_id in range(trajs_test.shape[0]):
            np.save(f'{folder_name}/test/data_{sample_id}.npy', trajs_test[sample_id])
        print(f'>> trajectories saved as {folder_name}/(train,val,test)/data_*.npy')

    print(f"\n{datetime.datetime.now().strftime('%a %b %d %H:%M:%S %Y')}")
    return 0

def generate_summary(data_folder, stimuli):

    """
    helper function to generate trajectory length for all subjects.
    writes to csv in output folder.
    Argvs:
    - data_folder: str, the data path that contains raw data files
    - output_folder: str, output path to which csv will be written.
    - stimuli: str, which stimuli to include
    """

    all_files = sorted(glob.glob(os.path.join(data_folder, f'Subject*_{stimuli}.csv')))

    data_summary = []
    for file in all_files:
        df = pd.read_csv(file)
        subject_id = os.path.basename(file).split('_')[0].lstrip('Subject')
        data_summary.append({'subject': int(subject_id),
                             'stimuli': stimuli,
                             'T': df.shape[0],
                             'ncol': df.shape[1]})

    df = pd.DataFrame(data_summary)
    df.sort_values(['subject', 'stimuli'], inplace=True)

    return df

def read_raw_data(data_folder, min_length, stimuli, nodes_list=None):
    """
    Read EEG raw data.
    The naming convention is Subject number and the two stimuli EC and EO, e.g. Subject3_EC.csv
    Argvs:
    - data_folder: str, the data path that contains raw data files
    - min_length: int, the minimum observation length for a subject to be included
    - stimuli: str, which stimuli to include
    - nodes_list: list, a list of nodes to be included. If none, read all nodes.

    Returns: a dictionary with key = subject_id and value = np.array of shape (T_obs, p) where p is the number of nodes
    """
    assert stimuli in ["EC", "EO"]
    all_files = sorted(glob.glob(os.path.join(data_folder, f'Subject*_{stimuli}.csv')))

    data_raw, dropped_subjects = {}, []
    for file in all_files:

        subject_id = os.path.basename(file).split('_')[0].lstrip('Subject')
        df = pd.read_csv(file)

        if df.shape[0] < min_length:
            print(f'# WARNING: Subject {subject_id} has trajectory of length {df.shape[0]} < minimum required {min_length}')
            dropped_subjects.append(subject_id)
            continue

        df.drop(['Unnamed: 0', 'times', 'epoch'], axis=1, inplace=True)

        if nodes_list:
            df = df[nodes_list]
        # use the first min_length observations
        df = df.head(min_length)
        data_raw[int(subject_id)] = df.to_numpy()

    # reorder subject id as continuous and keep the mapping
    order, data, subject_id_map = 0, {}, {}
    for subject_id, traj in data_raw.items():
        data[order] = traj
        subject_id_map[order] = subject_id
        order += 1

    print(f'>> effective number of subjects collected={len(data_raw)}; dropped_subjects={dropped_subjects}')
    return data, subject_id_map
    
def create_subsample(traj, sample_freq):
    """
    create subsamples by averaging over a window of length=sample_freq
    Argv:
    - traj: [T, p] where T is the number of observations and p the number of nodes
    Return:
    - ss_traj: [T//sample_freq, p]
    """
    ss_traj = np.empty((traj.shape[0]//sample_freq, traj.shape[1], sample_freq))
    for start_idx in range(sample_freq):
        ss_traj[:,:,start_idx] = traj[start_idx::sample_freq,:]
    ss_traj = np.mean(ss_traj, axis=-1)
    return ss_traj
    
if __name__ == "__main__":
    main()
