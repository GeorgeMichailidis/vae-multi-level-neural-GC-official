#!/usr/bin/env python3
"""
python train.py --ds_str='EEG_EC' --train_size=1000 --cuda=0
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import yaml

from utils import RealDataRunner, utils_logging

logger = utils_logging.get_logger()


def main(args):

    args.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    args.devices = [args.cuda] if args.accelerator == 'gpu' else 1

    ########################################
    ### step I: setup run environment
    ########################################
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
    
    for section_key, val in configs.items():
        setattr(args, section_key, val)

    ## add directory specification to args
    if not len(args.output_dir): ## no override
        suffix = args.network_params["model_type"]
        args.output_dir = os.path.join('output_real',f'{args.ds_str}-{suffix}-{args.train_size}')
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    ## create the corresponding directories
    if not os.path.exists(args.output_dir):
        logger.info(f'output_dir={args.output_dir} created')
    else:
        logger.warning(f'output_dir={args.output_dir} already exists; rm then recreated')
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    os.mkdir(args.ckpt_dir)
    
    args.ds_str = args.ds_str.split('_')[0]
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as handle:
        json.dump(vars(args), handle, default=str)
        logger.info(f"args saved to {os.path.join(args.output_dir, 'args.json')}")

    ########################################
    ### step II: initialize the runner, traing network
    ########################################
    runner = RealDataRunner(args)
    outDict = runner.end2end(eval_type='test')
    for key, outItem in outDict.items():
        save_path = os.path.join(args.output_dir, f'test_{key}.npy')
        np.save(save_path, outItem)
    
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ds_str', type=str, help='dataset to run',default='EEG_EC')
    parser.add_argument('--train_size', type=int, help='number of training samples',default=1000)
    parser.add_argument('--cuda', type=int, help='cuda device id',default=0)
    parser.add_argument('--run_seed',type=int, help='seed for the run; default to 0, corresponding to no override',default=0)
    parser.add_argument('--output_dir',type=str,help='override to default output folder',default='')
    parser.add_argument('--config',type=str, help='override to the config file used; default to empty',default='')
    args = parser.parse_args()
    
    print(f"python={'.'.join(map(str,sys.version_info[:3]))}; numpy={np.__version__}; torch={torch.__version__}")
    
    logger.info(f'START EXECUTION, SCRIPT={sys.argv[0]}')
    main(args)
    logger.info(f'DONE EXECUSION, SCRIPT={sys.argv[0]}')
