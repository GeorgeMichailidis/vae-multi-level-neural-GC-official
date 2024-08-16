#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import yaml

from utils import SimDataRunner, utils_logging

logger = utils_logging.get_logger()


def main(args):

    args.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    args.devices = [args.cuda] if args.accelerator == 'gpu' else 1

    ########################################
    ### step I: setup environment, sanity checks etc
    ########################################
    
    ## config setup
    if not len(args.config):
        args.config = os.path.join('configs', f'{args.ds_str}.yaml')
    logger.info('config_file={args.config}')

    with open(args.config) as f_ds:
        configs = yaml.safe_load(f_ds)
        
    if args.run_seed > 0:
        configs['train_params']['run_seed'] = args.run_seed
    if 'run_seed' in configs['train_params']:
        torch.manual_seed(configs['train_params']['run_seed'])
        torch.cuda.manual_seed(configs['train_params']['run_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"run_seed in use={configs['train_params']['run_seed']}")
    
    ## sanity check config specification
    _check_configs(configs)

    for section_key, val in configs.items():
        setattr(args, section_key, val)

    ## add directory specification to args
    if not len(args.output_dir): ## no override
        suffix = args.network_params["model_type"]
        args.output_dir = os.path.join('output_sim',f'{args.ds_str}_seed{args.seed}-{suffix}-{args.train_size}')
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    ## create the corresponding directories
    if not os.path.exists(args.output_dir):
        logger.info(f'output_dir={args.output_dir} created')
    else:
        logger.warning(f'output_dir={args.output_dir} already exists; rm then recreated')
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    os.mkdir(args.ckpt_dir)
    
    ## dump args.json for easier tracking and debugging
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as handle:
        json.dump(vars(args), handle, default=str)
        logger.info(f"args saved to {os.path.join(args.output_dir, 'args.json')}")
        
    ########################################
    ### step II: initialize the runner, traing network, perform evaluation
    ########################################

    ## instantiate the runner
    runner = SimDataRunner(args)
    #### train network and perform model evaluation on test set
    trainer = runner.train_network()
    ## perform evaluation on test set
    outDict = runner.model_eval(trainer, ckpt_type='last', file_type = 'test')
    for key, outItem in outDict.items():
        save_path = os.path.join(args.output_dir, f'test_{key}.npy')
        np.save(save_path, outItem)
        
    ## evaluate also on train set (but limited to test set size that is specified in the config)
    if not args.eval_test_only:
        outDict = runner.model_eval(trainer, ckpt_type='last', file_type = 'train', verbose = True)
        for key, outItem in outDict.items():
            save_path = os.path.join(args.output_dir, f'train_{key}.npy')
            np.save(save_path, outItem)

    return 0


def _check_configs(configs):
    """
    basic checks: entries in the configs are self-consistent
    """
    assert configs['DGP']['num_nodes'] == configs['data_params']['num_nodes']
    assert configs['DGP']['num_subjects'] == configs['data_params']['num_subjects']
    assert configs['DGP']['num_val'] >= configs['data_params']['val_size']
    assert configs['DGP']['num_test'] >= configs['data_params']['test_size']

    assert configs['data_params']['graph_type'] in ['binary','numeric']
    assert configs['network_params']['loss_type'] in ['MSE','NLL']
    
    assert configs['network_params']['model_type'] == 'simMultiSubVAE', f'unexpected model_type={configs["network_params"]["model_type"]}'
    for section_key, content in configs.items():
        if 'run_seed' in content.keys() and section_key != 'train_params':
            raise ValueError('run_seed must be specified under section train_params')
    
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ds_str', type=str, help='dataset to run',default='ds0')
    parser.add_argument('--train_size', type=int, help='number of training samples',default=1000)
    parser.add_argument('--cuda', type=int, help='cuda device id',default=0)
    parser.add_argument('--seed',type=int, help='seed for the dataset',default=0)
    parser.add_argument('--run_seed',type=int, help='seed for the run; default to 0, corresponding to no override',default=0)
    parser.add_argument('--output_dir',type=str,help='override to default output folder',default='')
    parser.add_argument('--config',type=str, help='override to the config file used; default to empty',default='')
    parser.add_argument('--eval-test-only',action='store_true',help='whether to only do evaluation on test data; default to false')
    args = parser.parse_args()
    
    print(f"python={'.'.join(map(str,sys.version_info[:3]))}; numpy={np.__version__}; torch={torch.__version__}")
    
    logger.info(f'START EXECUTION, SCRIPT={sys.argv[0]}')
    main(args)
    logger.info(f'DONE EXECUSION, SCRIPT={sys.argv[0]}')
