#!/usr/bin/env python3

"""
data generation according to Lorenz model and sample preparation
python generate_Lorenz.py --ds_str='Lorenz96' --seed=0
"""
import argparse
from pathlib import Path
import pickle
import os
import shutil
import sys
_ROOTDIR_ = str(Path(__file__).resolve().parents[1])
os.chdir(_ROOTDIR_)
sys.path.append(_ROOTDIR_)

import numpy as np
import yaml

from .simulators import simLorenz
from utils.utils_data import parse_trajectories, seq_to_samples
from utils import utils_logging

logger = utils_logging.get_logger()


def main(args):
    
    args.config = f'configs/{args.ds_str}.yaml'
    logger.info(f"config_file={args.config}")
    with open(args.config) as f:
        params = yaml.safe_load(f)['DGP']

    folder_name = f'data_sim/{args.ds_str}_seed{args.seed}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        logger.info(f'folder={folder_name} created')
    elif not args.sample_prep_only:
        logger.warning(f'folder={folder_name} existed; emptied and recreated')
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)

    args.params_loc = os.path.join(folder_name, 'params.pickle')
    args.trajs_loc = os.path.join(folder_name, 'trajs.pickle')

    if not args.sample_prep_only:
        logger.info(f'STAGE: SIMULATING TRAJECTORIES; DATASET NAME={args.ds_str}; SEED={args.seed}')

        ## set seed
        np.random.seed(args.seed)
        logger.info(f'random seed in use={args.seed}')
        
        ###########
        ## initialize the simulator and generate random forces
        ###########
        if params['dgp_str'] == 'Lorenz96':
            simulator = simLorenz(params)
        else:
            raise ValueError('unsupported dgp_str')

        simulator.gen_forces()

        ###########
        ## save down params used in the generation
        ###########
        with open(args.params_loc, 'wb') as handle:
            pickle.dump(simulator.Lorenz_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

        # same GC for all subjects
        gc_by_subject = np.stack([simulator.Lorenz_params['GC'] for i in range(params['num_subjects'])], axis=0)
        np.save(f'{folder_name}/GC_by_subject.npy', gc_by_subject)
        np.save(f'{folder_name}/GC_bar.npy', simulator.Lorenz_params['GC'])

        ###########
        ## generate and save down trajectories for each subject
        ###########
        trajs = simulator.sim_trajectory(T_obs=params['T_obs'], T_delta=params['T_delta'], sigma_obs=params['sigma_obs'])
        with open(args.trajs_loc, 'wb') as handle:
            pickle.dump(trajs, handle, protocol = pickle.HIGHEST_PROTOCOL)

        logger.info(f'simulated trajectories saved to {args.trajs_loc}')

    ###########
    ## parse trajectories and prep samples
    ###########
    logger.info(f'STAGE: PREPARING SAMPLES BASED ON SIMULATED TRAJECTORIES')
    with open(args.trajs_loc, 'rb') as handle:
        trajs = pickle.load(handle)
    print(f'>> trajectories loaded from {args.trajs_loc}')

    for folder_type in ['train','val','test']:
        folder_name_typed = os.path.join(folder_name, folder_type)
        if not os.path.exists(folder_name_typed):
            os.mkdir(folder_name_typed)
            logger.info(f'folder={folder_name_typed} created')
        else:
            logger.warning(f'folder={folder_name_typed} existed; emptied and recreated')
            shutil.rmtree(folder_name_typed)
            os.mkdir(folder_name_typed)

    num_val, num_test = params['num_val'], params['num_test']
    trajs_train, trajs_val, trajs_test = parse_trajectories(trajs,
                                                        num_val,
                                                        num_test,
                                                        context_length = params['context_length'],
                                                        prediction_length = params['target_length'],
                                                        stride= params['stride'],
                                                        verbose=args.verbose)

    ## save each sample as an .npy
    for sample_id in range(trajs_train.shape[0]):
        np.save(f'{folder_name}/train/data_{sample_id}.npy', trajs_train[sample_id])
    for sample_id in range(trajs_val.shape[0]):
        np.save(f'{folder_name}/val/data_{sample_id}.npy', trajs_val[sample_id])
    for sample_id in range(trajs_test.shape[0]):
        np.save(f'{folder_name}/test/data_{sample_id}.npy', trajs_test[sample_id])
    logger.info(f'parsed trajectories saved as {folder_name}/(train,val,test)/data_*.npy')
    
    print("============")
    print(f"To retrieve the Lorenz parameters used:")
    print(f"with open('{folder_name}/params.pickle','rb') as handle:")
    print(f"    params = pickle.load(handle)")
    print("============")
    
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ds_str', type=str, help='name for dataset to be generated',default='Lorenz96')
    parser.add_argument('--seed', type=int, help='seed value',default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--sample-prep-only', action='store_true')
    args = parser.parse_args()
    
    print(f"python={'.'.join(map(str,sys.version_info[:3]))}; numpy={np.__version__}")
    logger.info(f'START EXECUTION, SCRIPT={sys.argv[0]}')
    main(args)
    logger.info(f'DONE EXECUTION, SCRIPT={sys.argv[0]}')
