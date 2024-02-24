#!/usr/bin/env python3

"""
run real data experiments for a single subject using VAE-based method
python train_one.py --ds_str='Lorenz96' --train_size=1000 --cuda=0 --subject_id=0
"""

import os
import json
import numpy as np
import shutil
import yaml
import argparse
import torch
if float('.'.join(torch.__version__[:4].split('.')[:2])) >= 1.12:
    torch.set_float32_matmul_precision('high')

from utils import realRunner

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ds_str', type=str, help='dataset to run',default='ds0')
parser.add_argument('--train_size', type=int, help='number of training samples',default=1000)
parser.add_argument('--subject_id', type=int, help='id to the subject to run',default=0)
parser.add_argument('--cuda', type=int, help='cuda device id',default=0)
parser.add_argument('--run_seed',type=int, help='seed for the run; default to 0, corresponding to no override',default=0)
parser.add_argument('--output_dir',type=str,help='override to default output folder',default='')
parser.add_argument('--config',type=str, help='override to the config file used; default to empty',default='')

def main():

    global args
    args = parser.parse_args()
    
    setattr(args, 'train_size', int(args.train_size))
    setattr(args, 'accelerator', 'gpu' if torch.cuda.is_available() else 'cpu')
    setattr(args, 'devices', [args.cuda] if args.accelerator == 'gpu' else 1)
    
    ########################################
    ### step I: setup run environment
    ########################################
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
    
    for section_key, val in configs.items():
        setattr(args, section_key, val)

    ## directory setup
    if not len(args.output_dir): ## no override
        suffix = args.network_params["model_type"]
        setattr(args, 'output_dir', os.path.join('output_real',f'{args.ds_str}-{suffix}-{args.train_size}', f'subject_{args.subject_id}'))

    if not os.path.exists(args.output_dir):
        print(f'{args.output_dir} created')
    else:
        print(f'{args.output_dir} already exists; rm then recreated')
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    setattr(args, 'ckpt_dir', os.path.join(args.output_dir, 'ckpts'))
    os.mkdir(args.ckpt_dir)

    args.ds_str = args.ds_str.split('_')[0]
    with open(f'{args.output_dir}/args.json','w') as handle:
        json.dump(vars(args), handle)
    
    ########################################
    ### step II: initialize the runner, traing network
    ########################################
    
    runner = realRunner(args)
    outDict = runner.end2end(eval_type='test')
    for key, outItem in outDict.items():
        np.save(f'{args.output_dir}/test_{key}.npy', outItem)
    
    return 0

if __name__ == "__main__":
    main()
