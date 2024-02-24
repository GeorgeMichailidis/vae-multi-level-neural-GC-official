
"""
utility functions for plotting
"""

import json
import yaml
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def get_heatmap(mtx, axs_to_plot, rm_diag=False, threshold=None, threshold_by_quantile=False, vmin=None, vmax=None, alpha=1, labels=None, annot=False, color_palette='seismic'):
    
    cmap=sns.color_palette(color_palette, as_cmap=True)
    
    mtx = mtx.copy()
    if rm_diag:
        for i in range(mtx.shape[0]):
            mtx[i,i] = 0
        
    vmin = vmin or -1*np.max(np.abs(mtx))
    vmax = vmax or np.max(np.abs(mtx))
    
    if threshold is not None:
        if threshold_by_quantile:
            threshold = np.quantile(np.abs(mtx),threshold)
        mtx = mtx * 1 * (np.abs(mtx) > threshold)
        
    if labels is not None:
        mtx = pd.DataFrame(data=mtx, columns=labels, index=labels)
    
    g = sns.heatmap(mtx, linewidth=0.1, vmin=vmin, vmax=vmax, alpha=alpha,
            cmap=cmap,cbar=False, annot=annot, fmt=".2f", annot_kws={"fontsize":9},ax=axs_to_plot)
    g.set_yticklabels(g.get_yticklabels(), size=7, rotation=0)
    g.set_xticklabels(g.get_xticklabels(), size=7, rotation=0)
    for _, spine in axs_to_plot.spines.items():
        spine.set_visible(True)
    
    return

def get_heatmap_binary(mtx, axs_to_plot, threshold=0.01, rm_diag=False, alpha=1, color_palette='binary'):
    
    cmap = sns.color_palette(color_palette, as_cmap=True)
    bounds = [0.,threshold,1.]
    norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
    
    if rm_diag:
        mtx = mtx.copy()
        for i in range(mtx.shape[0]):
            mtx[i,i] = 0
    
    sns.heatmap(mtx, linewidth=0.1, cmap=cmap,norm=norm,cbar=False,
        annot=False, fmt=".2f", annot_kws={"fontsize":9},ax=axs_to_plot,alpha=alpha)
    for _, spine in axs_to_plot.spines.items():
        spine.set_visible(True)
    
    return

def get_heatmap_with_mask(mtx, mtx_bar, rm_diag=False, axs_to_plot=None, threshold=None,
                vmin=None, vmax=None, alpha=1, labels=None, annot=False, binary=False, color_palette='seismic'):

    cmap = sns.color_palette("binary", as_cmap=True) if binary else sns.color_palette(color_palette, as_cmap=True)
    cmap.set_bad('lightgray')

    mtx_mask = ((mtx!=0) & (mtx_bar!=0))
    
    mtx = mtx.copy()
    if binary:
        mtx = 1*(mtx != 0)
        vmin, vmax = None, None
    else:
        vmin = vmin or -1*np.max(np.abs(mtx))
        vmax = vmax or np.max(np.abs(mtx))
        if threshold is not None:
            mtx = mtx * 1 * (np.abs(mtx) > threshold * np.max(np.abs(mtx)))
            
    if rm_diag:
        for i in range(mtx.shape[0]):
            mtx[i,i] = 0
            mtx_mask[i,i] = 0
        
    if labels is not None:
        mtx = pd.DataFrame(data=mtx, columns=labels, index=labels)
    
    g = sns.heatmap(mtx, linewidths=0.01, vmin=vmin, vmax=vmax, alpha=alpha,
            cmap=cmap,cbar=False,mask=mtx_mask,
            annot=annot, fmt=".2f", annot_kws={"fontsize":9},ax=axs_to_plot)
    g.set_yticklabels(g.get_yticklabels(), size=7, rotation=0)
    g.set_xticklabels(g.get_xticklabels(), size=7, rotation=0)

    for _, spine in axs_to_plot.spines.items():
        spine.set_visible(True)
        
    return
