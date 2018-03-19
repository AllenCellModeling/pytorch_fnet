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

import scipy.misc

from sklearn.metrics import r2_score

from matplotlib import pyplot as plt

import glob
import pickle

import pdb

def evaluate_model(predictions_file = None, predictions_dir = None, path_save_dir='saved_models', save_error_maps=False, overwrite=True, reference_file = None):
    
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)        
    
    if predictions_file is not None:
        prediction_files = [predictions_file]
        save_dirs = [path_save_dir]
        
        train_or_tests = None
        structures = None
        
    else:
        #do a directory traversal
        prediction_files = glob.glob(predictions_dir + '/*/*/predictions.csv')
        save_dirs = [path.replace(predictions_dir, '') for path in prediction_files]
        save_dirs = [path.replace('predictions.csv', '') for path in save_dirs]
        save_dirs = [path_save_dir + os.sep + path for path in save_dirs]
        
        #do some jenky parsing
        split_on_filesep = [path.split('/') for path in  prediction_files]
        train_or_tests = [split[-2] for split in split_on_filesep]
        structures = [split[-3] for split in split_on_filesep]
        
#     pdb.set_trace()
        
            
    stats_file = path_save_dir + os.sep + 'stats.pkl' 
    if not overwrite and os.path.exists(stats_file):
        all_stats_list, stats_per_im_list = pickle.load( open( stats_file, "rb" ) )
    else:
        all_stats_list = list()
        stats_per_im_list = list()
        
        for prediction_file, structure, train_or_test, save_dir in zip(prediction_files, structures, train_or_tests, save_dirs):

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            pred_dir, _ = os.path.split(prediction_file)

            df_preds = pd.read_csv(prediction_file)

            #prepend the directory to the paths in the dataframe
            path_columns = [column for column in df_preds.columns if 'path' in column]
            for column in path_columns:
                not_nans = ~pd.isnull(df_preds[column])

                if np.any(not_nans):
                    df_preds[column][not_nans] = pred_dir + os.sep + df_preds[column][not_nans]


            df_preds['path_delta'] = [save_dir + os.sep + str(row[0]) + '_delta.tif' for row in df_preds.iterrows()]
            df_preds['path_stats'] = [save_dir + os.sep + str(row[0]) + '_stats.csv' for row in df_preds.iterrows()]

            path_stats_all = save_dir + os.sep + 'stats_all.csv'

            print('Working on ' + prediction_file)

            path_pred_col = [column for column in df_preds.columns if 'path_prediction' in column]
            if len(path_pred_col) == 0:
                path_pred_col = 'path_target'
            else:
                path_pred_col = path_pred_col[0]

            stats_per_im, stats_all = eval_images(df_preds['path_target'], 
                                        df_preds[path_pred_col], 
                                        df_preds['path_delta'], 
                                        df_preds['path_stats'], 
                                        path_stats_all)
            
            stats_per_im['structure'] = structure
            stats_per_im['train_or_test'] = train_or_test
            
            stats_all['structure'] = structure
            stats_all['train_or_test'] = train_or_test
            
            stats_per_im_list.append(stats_per_im)
            all_stats_list.append(stats_all)

        all_stats_list = pd.concat(all_stats_list)
        stats_per_im_list = pd.concat(stats_per_im_list)
            
        pickle.dump( [all_stats_list, stats_per_im_list] , open( stats_file, "wb" ) )
    
    stats_per_im_list['c_max'] = np.nan
    
    
    df_cmax = None
    if reference_file is not None:
        all_ref_stats_list, stats_ref_per_im_list = pickle.load( open( reference_file, "rb" ) )
        all_ref = all_ref_stats_list[all_ref_stats_list['train_or_test'] == 'train']
        all_ref_per_im = stats_ref_per_im_list[stats_ref_per_im_list['train_or_test'] == 'train']
        stats_per_im_list_train = stats_per_im_list[stats_per_im_list['train_or_test'] == 'train']
        
        u_structures = np.unique(all_ref['structure'])
    
        wildtypes = [structure for structure in u_structures if 'wildtype' in structure]
        
        wildtypes_short = np.array([wildtype.split('_')[1] for wildtype in wildtypes])
        
        vars_g = np.array([np.mean(all_ref[wildtype == all_ref['structure']]['var_target']) for wildtype in wildtypes])
            
        c_max_out = np.zeros(len(u_structures))
        noise_model = list()
        for structure, i in zip(u_structures, range(len(u_structures))):
            #set nan cmax values to the average post
            
            
            wt_map = [wildtype in structure for wildtype in wildtypes_short]
            if ~np.any(wt_map):
                wt_map = wildtypes_short == 'gfp'
            
            
            
            noise_model += wildtypes_short[wt_map].tolist()
            
            var_g = vars_g[wt_map]
            var_i = np.mean(all_ref_per_im['var_target'][all_ref_per_im['structure'] == structure])
            
            
            c_max_per_img = c_max(var_i, var_g) 
            c_max_out[i] = np.mean(c_max_per_img)
            
            
            struct_inds_ref = stats_ref_per_im_list['structure'] == structure
            struct_inds = stats_per_im_list['structure'] == structure
            cm = c_max(stats_ref_per_im_list['var_target'][struct_inds_ref], var_g)

            
            #if its wildtype gfp we know its undefined
            if structure == 'wildtype_gfp':
                cm = np.nan
            
            if np.sum(struct_inds) > 0:
                try:
                    stats_per_im_list['c_max'][struct_inds] = cm
                except:
                    pdb.set_trace()
            
#             var_i = all_ref_per_im['var_target'][all_ref_per_im['structure'] == structure]
            
#             r2_train = stats_per_im_list_train['r2'][stats_per_im_list_train['structure'] == structure]
#             r2 = np.corrcoef(r2_train, c_max_per_img)
            
#             if structure == 'desmoplakin':
#                 pdb.set_trace()
            
#             print(structure + ': ' + str(r2[0,1]))


        df_cmax = pd.DataFrame(np.stack([u_structures, noise_model, c_max_out]).T, columns=['structure', 'noise_model', 'c_max'])
        df_cmax.to_csv(path_save_dir + os.sep + 'c_max.csv')
       
    
    stats_per_im_list.to_csv(path_save_dir + os.sep + 'stats_per_im.csv')
                                
    fig_basename = path_save_dir + os.sep + 'stats_'
    
    filetypes = ['.eps', '.png']
    
    stats_to_print = ['r2']
    
    
    return all_stats_list, stats_per_im_list, df_cmax
    
#     for stat in stats_to_print:
#         for filetype in filetypes:
#             figure_save_path = fig_basename + stat + filetype
            

def c_max(var_i, var_g):
    cm = 1/np.sqrt(1 + (var_g / ((var_i-var_g)+1E-16)))
    
    return cm

def time_series_to_img(im_path_list, window_position = None, window_size = None, border_thickness = 0, im_save_path = None, border_color = 255):
    '''im_path_list is a list containing a list of images''' 
    
    im_list = []
    for im_t in im_path_list:
        channel_list = []
        for im_channel in im_t:
            im = imread(im_channel)
            
            if im.shape[1] == 1:
                im = im[:,0,:,:]
            
            if window_position is not None and window_size is not None:
                i_start = window_position[0]
                i_end = i_start+window_size[0]
                
                j_start = window_position[1]
                j_end = j_start+window_size[1]
                
                im_window = im[:, i_start:i_end,j_start:j_end]
            else:
                im_window = im
                
#             pdb.set_trace()
            channel_list+=[im_window]
            
            if border_thickness > 0:
                channel_list += [np.ones([1, border_thickness, im_window.shape[2]])*border_color]
            
            
        #these should all be channel depth of 1 or 3
        max_channel_depth = 1
        for channel in channel_list:
            if len(channel.shape) == 3 & channel.shape[0] != max_channel_depth:
                max_channel_depth = channel.shape[0]
                
        for channel, i in zip(channel_list, range(len(channel_list))):
            if len(channel.shape) < 3 or channel.shape[0] != max_channel_depth:
                channel_list[i] = np.tile(channel, [max_channel_depth, 1, 1])
       

        im_list += [np.concatenate(channel_list, 1)]

        if border_thickness > 0:
            im_list += [np.ones([max_channel_depth, im_list[-1].shape[1], border_thickness])*border_color]
        
    im_out = np.concatenate(im_list, 2)
    
    if im_save_path is not None:        
        scipy.misc.imsave(im_save_path, np.squeeze(im_out))

        
        
    return im_out
    
def stack_to_slices(im_path_list, window_position = None, window_size = None, border_thickness = 0, z_interval = [0,-1,1], im_save_path = None):
    '''im_path_list is a list containing a list of images, assume images are [c,y,x,z]''' 
    
    im_list = []
    for im_channel in im_path_list:
        channel_list = []
#         for im_channel in im_t:
        im_channel = np.squeeze(im_channel)

        im = im_channel[z_interval[0]:z_interval[1]:z_interval[2]]

        if window_position is not None and window_size is not None:
            i_start = window_position[0]
            i_end = i_start+window_size[0]

            j_start = window_position[1]
            j_end = j_start+window_size[1]

            im_window = im[:, i_start:i_end,j_start:j_end]
        else:
            im_window = im

        for z in im_window:
            channel_list+=[z]

            if border_thickness > 0:
                channel_list += [np.ones([im_window.shape[1], border_thickness])*255]
            
            
        pdb.set_trace()
                
        im_list += [np.concatenate(channel_list, 1)]

        if border_thickness > 0:
            im_list += [np.ones([max_channel_depth, im_list[-1].shape[1], border_thickness])*255]
        
        
        
    im_out = np.concatenate(im_list, 2)
    
    if im_save_path is not None:        
        scipy.misc.imsave(im_save_path, np.squeeze(im_out))

        
        
    return im_out    
    

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
    
def print_stats_all_v2(stats, figure_save_path, parameter_to_plot='r2', width = 0.34, fontsize = 8, figsize = (10,3), cmax_stats=None, show_train=True):
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    structures = stats['structure']
    u_structures = np.unique(structures)

    i = 0
    for structure in u_structures:

        
        
        struct_stats = stats[structure == stats['structure']]
        
        if show_train:
            train_or_test = ['train', 'test']
            colors = ['r', 'y']
            pos = i 
        else:
            train_or_test = ['test']
            colors = ['y']
            pos = i + width/2
            
        
        
        
        var_i = np.mean(struct_stats['var_target'])
       
        
        for group, color in zip(train_or_test, colors):
            
            group_stats = struct_stats[struct_stats['train_or_test'] == group]
            
            # pdb.set_trace()
            
            bplot = plt.boxplot(group_stats[parameter_to_plot], 0, '', positions = [pos], widths=[width], patch_artist=True, whis = 1.5)
            
            bplot['boxes'][0].set_facecolor(color)
            bplot['medians'][0].set_color('k')
            # pdb.set_trace()
            
            
            pos = pos + width
            
        # var_colors = ['c', 'm']
        
        if cmax_stats is not None:
            c_max = cmax_stats[cmax_stats['structure'] == structure]['c_max'].tolist()[0]
                
            plt.plot([i, i+width], [c_max]*2, color='k') 
            
    
        i += 1

        
    hlist = list()
    
    for group, color in zip(train_or_test, colors):
        h = mpl.patches.Patch(color=color, label=group)        
        hlist.append(h)
    
    leg = plt.legend(hlist, train_or_test, fontsize = fontsize,
                    loc=1,
                    borderaxespad=0,
                    frameon=False
                )
    
    ax.set_ylabel(parameter_to_plot)
    ax.set_xticks(np.arange(len(u_structures)) + width / 2)
    
    ax.set_xticklabels(np.array(u_structures))
    
    ax.set_xlim(-.5, len(u_structures))
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)  

    plt.savefig(figure_save_path, bbox_inches='tight')
    plt.close()    
    
def eval_images(path_targets, path_preds, path_save_delta, path_save_stats, path_save_stats_all):
    
    log_per_im = list()
    
    im_preds = list()
    im_targets = list()
    
    pbar = tqdm(zip(range(0, len(path_preds)), path_preds, path_targets, path_save_delta, path_save_stats))
    
    for i, path_pred, path_target, path_save_delta, path_save_stat in pbar:
        
        if pd.isnull(path_target):
            continue
        
        im_pred = imread(path_pred)
        im_target = imread(path_target)

        err_map, n_pixels, stats = get_stats(im_pred, im_target)
                
        stats['img'] = i
        
        im_preds.append(im_pred)
        im_targets.append(im_target)
        
        delta = im_pred - im_target
        
        df_per_im = pd.DataFrame.from_dict([stats])
        
        df_per_im.to_csv(path_save_stat)
        
        log_per_im.append(df_per_im)
        
    
    if len(log_per_im) == 0:
        return None, None
    else:
        log_per_im = pd.concat(log_per_im)
        
    im_pred_all_flat = np.hstack([im.flatten() for im in im_preds])
    im_target_all_flat = np.hstack([im.flatten() for im in im_targets])
    
    err_map, n_pixels, stats = get_stats(im_pred_all_flat, im_target_all_flat)

    
    # pdb.set_trace()
    log_all = pd.DataFrame.from_dict([stats])
    
    log_all.to_csv(path_save_stats_all)
    # stats_log
    

    
    return log_per_im, log_all
    