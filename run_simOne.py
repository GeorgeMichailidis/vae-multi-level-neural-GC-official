#!/usr/bin/env python3

import os
import json
import numpy as np
import shutil
import yaml
import argparse
import torch

from utils import simRunner

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ds_str', type=str, help='dataset to run',default='ds0')
parser.add_argument('--train_size', type=int, help='number of training samples',default=1000)
parser.add_argument('--subject_id', type=int, help='id to the subject to run',default=0)
parser.add_argument('--cuda', type=int, help='cuda device id',default=0)
parser.add_argument('--seed',type=int, help='seed for the dataset',default=0)
parser.add_argument('--run_seed',type=int, help='seed for the run; default to 0, corresponding to no override',default=0)
parser.add_argument('--output_dir',type=str,help='override to default output folder',default='')
parser.add_argument('--config',type=str, help='override to the config file used; default to empty',default='')
parser.add_argument('--eval-test-only',action='store_true',help='whether to only do evaluation on test data; default to false')

def main():

    global args
    args = parser.parse_args()
    
    setattr(args, 'train_size', int(args.train_size))
    setattr(args, 'accelerator', 'gpu' if torch.cuda.is_available() else 'cpu')
    setattr(args, 'devices', [args.cuda] if args.accelerator == 'gpu' else 1)
    
    ########################################
    ### step I: setup environment, sanity checks etc
    ########################################
    
    ## config setup
    if not len(args.config):
        setattr(args,'config', f'./configs/{args.ds_str}_oneSub.yaml')

    with open(args.config) as f_ds:
        configs = yaml.safe_load(f_ds)
        
    configs['data_params']['subject_id'] = args.subject_id
    if args.run_seed > 0:
        configs['train_params']['run_seed'] = args.run_seed
        
    if 'run_seed' in configs['train_params']:
        torch.manual_seed(configs['train_params']['run_seed'])
        torch.cuda.manual_seed(configs['train_params']['run_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    ## sanity check config specification
    _check_configs(configs)

    for section_key, val in configs.items():
        setattr(args, section_key, val)

    ## directory setup
    if not len(args.output_dir): ## no override
        suffix = args.network_params["model_type"]
        setattr(args, 'output_dir', os.path.join('output_sim',f'{args.ds_str}_seed{args.seed}-{suffix}-{args.train_size}', f'subject_{args.subject_id}'))

    if not os.path.exists(args.output_dir):
        print(f'{args.output_dir} created')
    else:
        print(f'{args.output_dir} already exists; rm then recreated')
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    setattr(args, 'ckpt_dir', os.path.join(args.output_dir, 'ckpts'))
    os.mkdir(args.ckpt_dir)

    
    with open(f'{args.output_dir}/args.json','w') as handle:
        json.dump(vars(args), handle)
    
    ########################################
    ### step II: initialize the runner, traing network, perform evaluation
    ########################################

    #### runner and model instantiation
    runner = simRunner(args)
    #### train network and perform model evaluation
    trainer = runner.train_network()

    ## perform evaluation on test set
    outDict = runner.model_eval(trainer, ckpt_type='last', file_type = 'test')
    for key, outItem in outDict.items():
        np.save(f'{args.output_dir}/test_{key}.npy', outItem)
    
    ## evaluate also on train set (but limited to test set size that is specified in the config)
    if not args.eval_test_only:
        outDict = runner.model_eval(trainer, ckpt_type='last', file_type = 'train', verbose = True)
        for key, outItem in outDict.items():
            np.save(f'{args.output_dir}/train_{key}.npy', outItem)

    return 0

def _check_configs(configs):
    """
    basic checks: entries in the configs are self-consistent
    """
    assert configs['data_params']['graph_type'] in ['binary','numeric']
    assert configs['network_params']['loss_type'] in ['MSE','NLL']
    
    assert configs['network_params']['model_type'] == 'simOneSubVAE'
    for section_key, content in configs.items():
        if 'run_seed' in content.keys() and section_key != 'train_params':
            raise ValueError('run_seed must be specified under section train_params')
    
    return


if __name__ == "__main__":
    main()
