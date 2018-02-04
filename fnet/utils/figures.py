

import argparse
import fnet.data
import fnet.fnet_model
from fnet.utils import delta2rgb, get_stats

import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import time
import warnings
from tqdm import tqdm

import matplotlib as mpl

from tifffile import imread, imsave

from sklearn.metrics import r2_score



from matplotlib import pyplot as plt


import pdb


def print_stats_all(stats_per_im, figure_save_path, parameter_to_plot='R2', width = 0.34, fontsize = 8, figsize = (10,3)):
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    u_structures = np.unique(stats_per_im['structure'])
    
    for index, row in stats_per_im.iterrows():
        
        pos = np.where(u_structures == row['structure'])[0]
        
        color = 'r'
        train_or_test = row['train_or_test']
        param = row[parameter_to_plot]
        
        if train_or_test == 'test':
            pos = pos + width
            color = 'y'
    
        ax.bar(pos, param, width, color=color) #, yerr=men_std)

    h1 = mpl.patches.Patch(color='r', label='train')        
    h2 = mpl.patches.Patch(color='y', label='test')

    leg = plt.legend([h1, h2], ['train', 'test'], fontsize = fontsize,
                    loc=1,
                    borderaxespad=0,
                    frameon=False
                )
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel(r'$R^2$')
    ax.set_xticks(np.arange(len(u_structures)) + width / 2)
    
    ax.set_xticklabels(np.array(u_structures))
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)  

    plt.savefig(figure_save_path, bbox_inches='tight')
    plt.close()
    
def print_stats_all_v2(stats, figure_save_path, parameter_to_plot='R2', width = 0.34, fontsize = 8, figsize = (10,3)):
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    u_structures = np.unique(stats['structure'])
    
    i = 0
    for structure in u_structures:

        struct_stats = stats[structure == stats['structure']]
        
        train_or_test = ['train', 'test']
        colors = ['r', 'y']
        
        pos = i 
        for group, color in zip(train_or_test, colors):
            
            group_stats = struct_stats[struct_stats['train_or_test'] == group]
            
            # pdb.set_trace()
            
            bplot = plt.boxplot(group_stats[parameter_to_plot], 0, '', positions = [pos], widths=[width], patch_artist=True)
            
            bplot['boxes'][0].set_facecolor(color)
            bplot['medians'][0].set_color('k')
            # pdb.set_trace()
            
            pos = pos + width
            
        
        i += 1
#         color = 'r'
#         train_or_test = row['train_or_test']
#         param = row[parameter_to_plot]
        
#         if train_or_test == 'test':
#             pos = pos + width
#             color = 'y'
    
    hlist = list()
    
    for group, color in zip(train_or_test, colors):
        h = mpl.patches.Patch(color=color, label=group)        
        hlist.append(h)
    
    leg = plt.legend(hlist, ['train', 'test'], fontsize = fontsize,
                    loc=1,
                    borderaxespad=0,
                    frameon=False
                )
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel(r'$R^2$')
    ax.set_xticks(np.arange(len(u_structures)) + width / 2)
    
    ax.set_xticklabels(np.array(u_structures))
    
    ax.set_xlim(-.5, len(u_structures))
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)  

    plt.savefig(figure_save_path, bbox_inches='tight')
    plt.close()    
    
def eval_images(path_targets, path_preds, path_save_delta, path_save_stats, path_save_stats_all):
    cols = ['img', 'SE', 'MSE', 'delta_min', 'delta_max', 'n_pixels', 'R2']
    
    log_per_im = list()
    
    im_preds = list()
    im_targets = list()
    
    pbar = tqdm(zip(range(0, len(path_preds)), path_preds, path_targets, path_save_delta, path_save_stats))
    
    for i, path_pred, path_target, path_save_delta, path_save_stat in pbar:
        
        if pd.isnull(path_target):
            continue
        
        im_pred = imread(path_pred)
        im_target = imread(path_target)

        err_map, n_pixels, se, mse, r2, delta_min, delta_max, percentiles = get_stats(im_pred, im_target)
                
            
        
        im_preds.append(im_pred)
        im_targets.append(im_target)
        
        delta = im_pred - im_target
        

        imsave(path_save_delta, delta)
        
        #['SE', 'MSE', 'delta_min', 'delta_max', 'n_pixels', 'R2']  
        
    
        prct_cols = ['prct_' + str(k) for k in list(percentiles.keys())]
        prcts = list(percentiles.values())
        
        df_per_im = pd.DataFrame([[i, se, mse, delta_min, delta_max, n_pixels, r2]+prcts], columns = cols+prct_cols)
        
        df_per_im.to_csv(path_save_stat)
        
        log_per_im.append(df_per_im)
        
    
    if len(log_per_im) == 0:
        return None, None
    else:
        log_per_im = pd.concat(log_per_im)
        
    im_pred_all_flat = np.hstack([im.flatten() for im in im_preds])
    im_target_all_flat = np.hstack([im.flatten() for im in im_targets])
    
    err_map, n_pixels, se, mse, r2, delta_min, delta_max, percentiles = get_stats(im_pred_all_flat, im_target_all_flat)
    
    # pdb.set_trace()

    prct_cols = ['prct_' + str(k) for k in list(percentiles.keys())]
    prcts = list(percentiles.values())

    log_all = pd.DataFrame([[se, mse, delta_min, delta_max, n_pixels, r2] + prcts], columns = cols[1:]+prct_cols)
    log_all.to_csv(path_save_stats_all)
    # stats_log
    
    return log_per_im, log_all
    