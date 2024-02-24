"""
script for running performance check of an experiment
python eval_sim.py --ds_str=LinearVAR --data_seed=0
"""

import os
import sys
import yaml
import glob
import datetime
import argparse
import numpy as np

from utils.utils_eval import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--ds_str', type=str, help='dataset for which performance will be checked',default='ds0')
parser.add_argument('--seed', type=int, help='data seed',default=0)
parser.add_argument('--logname', type=str, help='overriding evaluation logname; default to None')
parser.add_argument('--config', type=str, help='overriding config loc; default to None')
parser.add_argument('--output_loc', type=str, help='overriding location for output; default to None')
parser.add_argument('--data_loc', type=str, help='overriding location for data; default to None')

#################
## usage: to evaluate results from specific folders: add folder name here (but without the output_sim/)
## otherwise default will be used and the code will automatically glob folders matching specific patterns
#################
_METHODS = {
    'simMultiSubVAE': [],
    'simOneSubVAE': [],
}

def main():

    global args
    args = parser.parse_args()
    
    if args.data_loc is None:
        args.data_loc = f'data_sim/{args.ds_str}_seed{args.seed}'
    if args.output_loc is None:
        args.output_loc = 'output_sim'
    
    if args.config is None:
        args.config = f'configs/{args.ds_str}.yaml'
    with open(args.config) as handle:
        configs = yaml.safe_load(handle)
        
    args.num_nodes = configs['data_params']['num_nodes']
    args.num_subjects = configs['data_params']['num_subjects']
    args.graph_key = configs['data_params'].get('graph_key','graph')
    args.include_diag = configs['data_params'].get('include_diag',True)
    args.metrics = ['acc','f1'] if args.ds_str == 'Springs5' else ['auroc','auprc','f1best']
    
    ## set identifier for this eval
    if args.logname is None:
        args.logname = f'{args.ds_str}_{args.seed}_eval'
    if not os.path.exists('logs'):
        os.mkdir('logs')
        
    print(f'Evaluation results will be redirected to logs/{args.logname}.log')
    
    sys.stdout = open(f'logs/{args.logname}.log', 'w')
    
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'METHODS in eval = {list(_METHODS.keys())}')
    print(f'=========')
    print(args)
    print(f'=========')
    
    ## evaluate each method
    for method_key, folders in _METHODS.items():
        if not len(folders): ## evaluation folder not specified: use the default
            folders = glob.glob(f'{args.output_loc}/{args.ds_str}_seed{args.seed}-{method_key}-*')
            if len(folders) == 0:
                folders = glob.glob(f'{args.output_loc}/{args.ds_str}/{args.ds_str}_seed{args.seed}-{method_key}-*')
                folders = sorted([fname.replace(f'{args.output_loc}/','') for fname in folders])
            else:
                folders = sorted([fname.replace(f'{args.output_loc}/','') for fname in folders])
            
            print(f'\n##################')
            print(f'##')
            print(f'## method={method_key}; folders={folders}')
            print(f'##')
            print(f'##################')
                
        for output_folder in folders:
            try:
                subj_eval, bar_eval = eval_vaeGraphs(output_folder = f'{args.output_loc}/{output_folder}',
                                                     data_folder = args.data_loc,
                                                     config_file = args.config)
                                                     
                print(f'\n[[output_folder={output_folder}]]')
                ## print out the evaluation results
                for metric_idx, curr_metric in enumerate(args.metrics):
                    curr_metric_by_subj = subj_eval[curr_metric]
                    eval_by_subject_str = ','.join(["{:.1f}%".format(x*100) for x in curr_metric_by_subj])
                    if metric_idx > 0:
                        print(f'----------')
                    print(f'>> avg{curr_metric.upper()}={np.mean(curr_metric_by_subj)*100:.1f}% ({np.std(curr_metric_by_subj)*100:.2f}%): [{eval_by_subject_str}]')
                    print(f'>> bar{curr_metric.upper()}={bar_eval[curr_metric]*100:.1f}%')
            except Exception as e:
                print(f'some error occurred while processing {output_folder}: {str(e)}')
            
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    return 0
    
################################
##
## helper functions for evaluating each individual method
##
################################

def eval_simVAE(output_folder, data_loc, config):
    subj_eval, bar_eval = eval_vaeGraphs(output_folder, data_folder=data_loc, config_file=config)
    return subj_eval, bar_eval

## entry point
if __name__ == '__main__':
    main()
