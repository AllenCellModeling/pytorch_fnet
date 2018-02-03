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

from tifffile import imread, imsave

from sklearn.metrics import r2_score


import pdb


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
    