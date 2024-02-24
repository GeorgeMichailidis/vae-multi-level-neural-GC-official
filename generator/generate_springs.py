#!/usr/bin/env python3

"""
data generation according for the Springs
python generate_springs.py --ds_str='Springs5' --seed=0 --mp_cores=2
"""

from pathlib import Path
_ROOTDIR_ = str(Path(__file__).resolve().parents[1])
import os
import sys
sys.path.append(_ROOTDIR_)
os.chdir(_ROOTDIR_)

import argparse
import yaml
import numpy as np
import multiprocessing
import shutil
import pickle
import datetime
import importlib

from generator import simSprings

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ds_str', type=str, help='name for dataset to be generated',default='VAR1')
parser.add_argument('--seed', type=int, help='seed value',default=0)
parser.add_argument('--mp_cores', type=int, help='number of cores used for mp',default=1)
parser.add_argument('--verbose', action='store_true',help='default to false')

def main():
    
    global args
    args = parser.parse_args()
    setattr(args, 'config', f'configs/{args.ds_str}.yaml')
    
    with open(args.config) as f:
        params = yaml.safe_load(f)['DGP']
    
    folder_name = f'data_sim/{args.ds_str}_seed{args.seed}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'folder={folder_name} created')
    else:
        print(f'folder={folder_name} existed; emptied and recreated')
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)
    
    print(f"{datetime.datetime.now().strftime('%a %b %d %H:%M:%S ET %Y')}\n")
    
    print(f"python={'.'.join(map(str,sys.version_info[:3]))}")
    print(f"numpy={np.__version__}")
    print(f"config_file={args.config}",end='\n\n')
    
    print(f'##### [{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] SIMULATING TRAJECTORIES; DATASET NAME={args.ds_str}; SEED={args.seed}')

    ## set seed
    np.random.seed(args.seed)
    
    ###########
    ## initialize the simulator and generate the graphs
    ###########
    simulator = simSprings(params)
    
    # graph_bar
    if not params.get('degenerate_beta',False):
        graph_bar = simulator.generate_edges_from_beta(alpha=1,beta=1)
    else:
        graph_bar = 0.5 * np.ones((params['num_nodes'],params['num_nodes']))
    
    np.save(f'{folder_name}/graph_bar.npy', graph_bar)
    print(f'>> graph_bar (degenerate={params.get("degenerate_beta",False)}) saved as {folder_name}/graph_bar.npy, shape={graph_bar.shape}')
    
    # extract the upper triangular as the probability for subject-level graphs
    one_prob = graph_bar[np.tril_indices(params['num_nodes'],-1)]
    
    if params.get('deterministic_graph',True):
        graph_by_subject = np.empty((params['num_subjects'], params['num_nodes'], params['num_nodes']))
        for subject_id in range(params['num_subjects']):
            graph_by_subject[subject_id, :, :] = simulator.generate_edges_from_prob(one_prob)
    else:
        graph_by_subject = [None] * params['num_subjects']
        
    ###########
    ## generate trajectories based on graphs
    ###########
    
    T_save = int(params['T_obs']/params['sample_freq']-1)
    
    global sample_traj, sample_graph
    
    for data_type in ['train','val','test']:
        print(f'##### [{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] SIMULATING TRAJECTORIES FOR {data_type}')
        if args.mp_cores == 1:
            
            sample_traj = np.empty((params[f'num_{data_type}'], params['num_subjects'], T_save, params['num_nodes'], 4))
            sample_graph = np.empty((params[f'num_{data_type}'], params['num_subjects'], params['num_nodes'], params['num_nodes']))
            
            for subject_id in range(params['num_subjects']):
                print(f'===== subject_id={subject_id} =====')
                sample_traj[:,subject_id,:,:,:], sample_graph[:,subject_id,:,:] = simulator.generate_dataset_one_subject(num_samples = params[f'num_{data_type}'],T_obs = params['T_obs'],sample_freq = params['sample_freq'], one_prob=one_prob, fix_edges= graph_by_subject[subject_id], seed = int(100*args.seed + subject_id))
        else:
            
            sample_traj, sample_graph = [], []
            pool = multiprocessing.Pool(processes = args.mp_cores)
            for subject_id in range(params['num_subjects']):
                print(f'===== subject_id={subject_id} =====')
                pool.apply_async(simulator.generate_dataset_one_subject,
                                 args=(params[f'num_{data_type}'], params['T_obs'], params['sample_freq'], one_prob, graph_by_subject[subject_id], int(100*args.seed + subject_id)),
                                 callback = _add_sim_to_collection)
            pool.close()
            pool.join()
            
            sample_traj, sample_graph = np.stack(sample_traj, axis=1), np.stack(sample_graph,axis=1)
        
        ## save down the subject-specific graph based on the sample_graph to ensure the correct mapping between trajectory and graph
        ## this doesn't matter for single-core processing, but in the case of multi-processing orders are no longer preserved !!
        if params.get('deterministic_graph',True):
            graph_by_subject = np.mean(sample_graph,axis=0)
            np.save(f'{folder_name}/graph_by_subject.npy', graph_by_subject)
            print(f'>> graph_by_subject saved as {folder_name}/graph_by_subject.npy, shape={graph_by_subject.shape}')
                
        data_folder = os.path.join(folder_name, data_type)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
            
        ## save by sample
        for sample_id in range(params[f'num_{data_type}']):
            np.save(f'{data_folder}/data_{sample_id}.npy', sample_traj[sample_id])
            np.save(f'{data_folder}/graph_{sample_id}.npy', sample_graph[sample_id])
        
        print(f'trajectories saved as {data_folder}/data_*.npy; graphs saved as {data_folder}/graph_*.npy')
    
    print(f"\n{datetime.datetime.now().strftime('%a %b %d %H:%M:%S ET %Y')}")
    
    return 0
    
def _add_sim_to_collection(sim):

    sample_traj.append(sim[0])
    sample_graph.append(sim[1])

if __name__ == "__main__":
    main()
