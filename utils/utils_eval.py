
"""
utility functions for gathering, processing and evaluating results
"""
import json
import yaml
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

## set to true to fix the sign in-determininancy across subjects
## this is primarily for metrics calculation so that at the time of averaging things don't incorrectly cancel out
## for plotting, set it to False
_OneSub_UNSIGNED_ = True

###########################
## result normalization
###########################

def extract_indices(n, include_diag=True):
    """ helper function for determine the indices """
    valid_loc = np.ones([n, n]) if include_diag else np.ones([n, n]) - np.eye(n)
    return  np.where(valid_loc)

def normalize_and_reshape(x, n=20, include_diag=True, normalize=True):
    """
    Argv:
    - x: [sample_size, num_edges] or [sample_size, 1, num_edges] for the output of single subject
    - n: number of nodes
    Return: [num_nodes, num_nodes]
    """
    if x.ndim == 3:
        assert x.shape[1] == 1
        x = np.squeeze(x, axis=1)
    graph = np.mean(x, axis=0) ## [num_edges,]
    if normalize:
        graph = graph/np.max(np.abs(graph))
    
    graph_new = np.zeros((n,n))
    indices = extract_indices(n, include_diag=include_diag)
    graph_new[indices[0],indices[1]] = graph
    return graph_new

def normalize_and_reshape_by_subject(x, n=20, include_diag=True, normalize=True):
    """
    perform normalize and reshape for all subjects, assuming the estimates stack at the 1st dimension
    Argv
    - x: [sample_size, num_subjects, num_edges]
    - n: number of nodes
    Return: [num_subjects, num_nodes, num_nodes]
    """
    graph_by_subject = []
    for subject_id in range(x.shape[1]):
        graph = normalize_and_reshape(x[:,subject_id,:],n=n, include_diag=include_diag, normalize=normalize)
        graph_by_subject.append(graph)
    return np.stack(graph_by_subject,axis=0)

###########################
## result gathering + normalization
###########################

def gather_vaeGraphs(output_folder, n_nodes, n_subjects, runType='multi', include_diag=True, normalize=True):
    """
    gather estimated graphs for VAE-based methods
    Return: a tuple of [num_subjects, num_nodes, num_nodes], [num_nodes, num_nodes]
    """
    if runType == 'multi':
    
        estGC_multi_raw = np.load(f'{output_folder}/test_subjGraphs.npy')
        assert estGC_multi_raw.shape[1] == n_subjects, f'expecting {n_subjects}, getting raw results for {estGC_multi_raw.shape[1]} subjects'
        
        estGC = normalize_and_reshape_by_subject(estGC_multi_raw, n=n_nodes, include_diag=include_diag, normalize=True)
        
        estGC_bar_multi = np.load(f'{output_folder}/test_barGraphs.npy')
        estGC_bar = normalize_and_reshape(estGC_bar_multi, n=n_nodes, include_diag=include_diag, normalize=True)
    
    else: ## runType=='one':
    
        estGC_one_raw = []
        for subject_id in range(n_subjects):
            estGC = np.load(f'{output_folder}/subject_{subject_id}/test_subjGraphs.npy')
            
            ## for fixing the sign indeterminancy so that when averaging across subjects, entries are not cancelling out
            if 'Lokta' not in output_folder:
                if subject_id == 0:
                    sign_ref = np.sign(estGC[0,0,0])
                if np.sign(estGC[0,0,0]) != sign_ref:
                    estGC = -1 * estGC
            else:
                if subject_id == 0:
                    sign_ref = np.sign(estGC[0,0,10])
                if np.sign(estGC[0,0,10]) != sign_ref:
                    estGC = -1 * estGC
                
            estGC_one_raw.append(estGC.squeeze(1))
        
        estGC_one_raw = np.stack(estGC_one_raw,axis=1)
        estGC = normalize_and_reshape_by_subject(estGC_one_raw, n=n_nodes, include_diag=include_diag, normalize=True)
        if _OneSub_UNSIGNED_:
            estGC_bar_one = np.mean(np.abs(estGC_one_raw),axis=1)
        else:
            estGC_bar_one = np.mean(estGC_one_raw,axis=1)
        
        estGC_bar = normalize_and_reshape(estGC_bar_one, n=n_nodes, include_diag=include_diag, normalize=True)
    
    return estGC, estGC_bar

###########################
## gather metric suite given threshold
###########################
def gather_metrics(true_graph, est_graph, include_diag=True, threshold=None):
    
    indices = extract_indices(true_graph.shape[-1], include_diag=include_diag)
    y_true = 1 * (np.abs(true_graph[indices[0],indices[1]]) != 0)
    y_pred = est_graph[indices[0],indices[1]]
    if threshold:
        y_pred = 1 * (np.abs(y_pred) > threshold)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metric_suite = {}
    
    metric_suite['specificity'] = tn/(tn+fp)
    metric_suite['sensitivity'] = tp/(tp+fn)
    metric_suite['precision']= tp/(tp+fp)
    metric_suite['recall']= tp/(tp+fn)

    metric_suite['f1score'] = 2*tp/(2*tp+fp+fn)
    metric_suite['mcc'] = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    metric_suite['acc'] = (tp+tn)/(tp+tn+fp+fn)
    metric_suite['ba'] = (metric_suite['specificity']+metric_suite['sensitivity'])/2
    
    return metric_suite

def gather_metrics_by_subject(true_by_subject, est_by_subject, include_diag=True, threshold=None):
    
    metric_suite_arr = []
    for subject_id in range(est_by_subject.shape[0]):
        metric_suite = gather_metrics(true_by_subject[subject_id], est_by_subject[subject_id], include_diag=include_diag, threshold=threshold)
        metric_suite_arr.append(metric_suite)
    return metric_suite_arr

###########################
## get F1 score
###########################
def get_f1_arr(true_graph, est_graph, include_diag=True, thresholds=np.arange(0.01,1,0.01)):
    """
    - true_graph/est_graph: [num_nodes, num_nodes]
    Return: f1 score array
    """
    f1_score_arr = []
    for threshold in thresholds:
        metric_suite = gather_metrics(true_graph, est_graph, include_diag=include_diag, threshold=threshold)
        f1_score_arr.append(metric_suite['f1score'])
        
    f1_score = np.array(f1_score_arr)
    f1_best = np.max(f1_score_arr)
    thresh_best = thresholds[np.argmax(f1_score_arr)]
    
    return f1_score_arr, f1_best, thresh_best

def get_f1_best_by_subject(true_by_subject, est_by_subject, include_diag=True, thresholds=np.arange(0.01,1,0.01)):
    """
    true_by_subject, est_by_subject: [num_subjects, num_nodes, num_nodes]
    Return: two lists, the first corresponding to the array, the second corresponding to the best f1 score
    """
    f1_best_by_subj, f1_score_arr_by_subj = [], []
    
    for subject_id in range(est_by_subject.shape[0]):
        f1_score_arr, f1_best, _ = get_f1_arr(true_by_subject[subject_id], est_by_subject[subject_id], include_diag=include_diag, thresholds=thresholds)
        f1_best_by_subj.append(f1_best)
        f1_score_arr_by_subj.append(f1_score_arr)

    return f1_score_arr_by_subj, f1_best_by_subj

###########################
## get ACC
###########################
def get_acc(true_graph, est_graph, include_diag=True, threshold=0.5):
    """
    - true_graph/est_graph: [num_nodes, num_nodes]
    Return: accuracy
    """
    indices = extract_indices(true_graph.shape[-1], include_diag=include_diag)
    y_true = 1 * (np.abs(true_graph[indices[0],indices[1]]) >= threshold)
    y_pred = 1 * (np.abs(est_graph[indices[0],indices[1]]) >= threshold)
    
    return accuracy_score(y_true, y_pred)

def get_acc_by_subject(true_by_subject, est_by_subject, include_diag=True, threshold=0.5):
    """
    true_by_subject, est_by_subject: [num_subjects, num_nodes, num_nodes]
    Return: a list recording the acc for each subject
    """
    acc = []
    for subject_id in range(est_by_subject.shape[0]):
        acc_val = get_acc(true_by_subject[subject_id], est_by_subject[subject_id], include_diag=include_diag, threshold=threshold)
        acc.append(acc_val)
    return acc

###########################
## get TPR, TNR (roc curve)
###########################
def get_tpr_tnr(true_graph, est_graph, include_diag=True, grid=np.arange(0.01,1,0.01), use_default_auroc=True):
    """
    - true_graph/est_graph: [num_nodes, num_nodes]
    Return: a tuple of (tptn: numpy.array with col 0 tpr and col 1 tnr, auroc: float)
    """
    indices = extract_indices(true_graph.shape[-1], include_diag=include_diag)
    y_true = 1 * (np.abs(true_graph[indices[0],indices[1]]) != 0)
    
    tpr, tnr = [], []
    for threshold in grid:
        metric_suite = gather_metrics(true_graph, est_graph, include_diag=include_diag, threshold=threshold)
        tpr.append(metric_suite['sensitivity'])
        tnr.append(metric_suite['specificity'])

    tpr, tnr = np.array(tpr), np.array(tnr)
    tptn = np.stack([tpr,tnr],axis=1)
    
    if not use_default_auroc:
        auroc = metrics.auc(1-tptn[:,1], tptn[:,0])
    else:
        auroc = metrics.roc_auc_score(y_true, np.abs(est_graph[indices[0],indices[1]]))
    
    return tptn, auroc

def get_tpr_tnr_by_subject(true_by_subject, est_by_subject, include_diag=True, grid=np.arange(0.01,1,0.01), use_default_auroc=True):
    """
    true_by_subject, est_by_subject: [num_subjects, num_nodes, num_nodes]
    Return: a tuple of two lists
    """
    tptn, auroc = [], []
    for subject_id in range(est_by_subject.shape[0]):
        tptn_arr, auroc_val = get_tpr_tnr(true_by_subject[subject_id],
                                        est_by_subject[subject_id],
                                        include_diag=include_diag,
                                        grid=grid,
                                        use_default_auroc=use_default_auroc)
        tptn.append(tptn_arr)
        auroc.append(auroc_val)
    return tptn, auroc

###########################
## get PRC, RECALL curve
###########################
def get_prc_rec(true_graph, est_graph, include_diag=True, grid=np.arange(0.01,1,0.01), use_default_ap=True):
    """
    - true_graph/est_graph: [num_nodes, num_nodes]
    Return: a tuple of (prcrec: numpy.array with col 0 precision and col 1 recall, auprc: float)
    """
    indices = extract_indices(true_graph.shape[-1], include_diag=include_diag)
    y_true = 1 * (np.abs(true_graph[indices[0],indices[1]]) != 0)
    
    prc, rec = [], []
    for threshold in grid:
        metric_suite = gather_metrics(true_graph, est_graph, include_diag=include_diag, threshold=threshold)
        prc.append(metric_suite['precision'])
        rec.append(metric_suite['recall'])

    prc, rec = np.array(prc), np.array(rec)
    prcrec = np.stack([prc,rec],axis=1)
    
    if not use_default_ap:
        auprc = metrics.auc(prcrec[:,1], prcrec[:,0])
    else:
        auprc = metrics.average_precision_score(y_true, np.abs(est_graph[indices[0],indices[1]]))
    
    return prcrec, auprc

def get_prc_rec_by_subject(true_by_subject, est_by_subject, include_diag=True, grid=np.arange(0.01,1,0.01), use_default_ap=True):
    """
    true_by_subject, est_by_subject: [num_subjects, num_nodes, num_nodes]
    Return: a tuple of two lists
    """
    prcrec, auprc = [], []
    for subject_id in range(est_by_subject.shape[0]):
        prcrec_arr, auprc_val = get_prc_rec(true_by_subject[subject_id],
                                            est_by_subject[subject_id],
                                            include_diag=include_diag,
                                            grid=grid,
                                            use_default_ap=use_default_ap)
        prcrec.append(prcrec_arr)
        auprc.append(auprc_val)
    return prcrec, auprc

###########################
## get Frobenius norm error
###########################
def get_errF(true_graph, est_graph, include_diag=True):

    indices = extract_indices(true_graph.shape[-1], include_diag=include_diag)
    y_true = true_graph[indices[0],indices[1]]
    y_pred = est_graph[indices[0],indices[1]]
    
    return np.linalg.norm(y_true - y_pred)
    
###########################
## main functions to perform evaluation
###########################
def eval_vaeGraphs(output_folder, data_folder=None, config_file=None):
    """
    gather results and perform evaluation for vae-based methods
    """
    try:
        with open(f'{output_folder}/args.json') as f:
            args = json.load(f)
        assert args["network_params"]["model_type"] == 'simMultiSubVAE', 'incorrect model_type'
        runType = 'multi'
    except FileNotFoundError:
        with open(f'{output_folder}/subject_0/args.json') as f:
            args = json.load(f)
        runType = 'one'
    except Exception as e:
        raise Exception(f'cannot load args; {str(e)}')
        
    if config_file is None:
        config_file = f"configs/{args['ds_str']}.yaml"
    with open(config_file) as f_ds:
        configs = yaml.safe_load(f_ds)
        
    n_nodes = configs["DGP"]["num_nodes"]
    n_subjects = configs["DGP"]["num_subjects"]
    graph_key = configs["data_params"].get("graph_key",'graph')
    include_diag = configs["data_params"].get("include_diag",True)
    
    if data_folder is None:
        data_folder = f'data_sim/{args["ds_str"]}_seed{args["seed"]}'
    
    GC_bar, GC_by_subj = np.load(f'{data_folder}/{graph_key}_bar.npy'), np.load(f'{data_folder}/{graph_key}_by_subject.npy')

    estGC, estGC_bar = gather_vaeGraphs(output_folder, n_nodes, n_subjects, runType, include_diag, normalize=True)
    
    if args['ds_str'] == 'Springs5':
    
        metrics_by_subj = gather_metrics_by_subject(GC_by_subj, estGC, include_diag=include_diag, threshold=0.5)
        acc_by_subj, f1_by_subj = [x['acc'] for x in metrics_by_subj], [x['f1score'] for x in metrics_by_subj]
        
        ferror_bar = get_errF(GC_bar, estGC_bar, include_diag=include_diag)
        
        return {'est':estGC, 'truth':GC_by_subj, 'acc':acc_by_subj, 'f1': f1_by_subj}, {'est':estGC_bar, 'truth':GC_bar, 'acc': ferror_bar, 'f1': np.nan}
    
    else:
        tptn_by_subj, auroc_by_subj = get_tpr_tnr_by_subject(GC_by_subj, estGC, include_diag=include_diag)
        tptn_bar, auroc_bar = get_tpr_tnr(GC_bar, estGC_bar, include_diag=include_diag)
        
        prcrec_by_subj, auprc_by_subj = get_prc_rec_by_subject(GC_by_subj, estGC, include_diag=include_diag)
        prcrec_bar, auprc_bar = get_prc_rec(GC_bar, estGC_bar, include_diag=include_diag)
        
        f1_by_subj, f1best_by_subj = get_f1_best_by_subject(GC_by_subj, estGC, include_diag=include_diag)
        f1_bar, f1best_bar, _ = get_f1_arr(GC_bar, estGC_bar, include_diag=include_diag)
        
        return {'est':estGC, 'truth':GC_by_subj, 'tptn':tptn_by_subj, 'auroc':auroc_by_subj, 'prcrec':prcrec_by_subj, 'auprc':auprc_by_subj, 'f1':f1_by_subj, 'f1best':f1best_by_subj}, {'est':estGC_bar, 'truth':GC_bar, 'tptn':tptn_bar, 'auroc':auroc_bar, 'prcrec':prcrec_bar, 'auprc':auprc_bar, 'f1': f1_bar, 'f1best':f1best_bar}
