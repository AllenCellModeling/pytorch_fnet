import argparse

import matplotlib as mpl
mpl.use('Agg')

from fnet.utils.figures import eval_images, print_stats_all, print_stats_all_v2

import logging
import os
import pandas as pd
import sys
import warnings

import numpy as np
import glob

import pickle


import pdb

    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file', default=None, help='two column .csv of paths to path_prediction and path_target')
    parser.add_argument('--predictions_dir', default=None, help='project directory. Mutually exclusive of predictions_dir')
    parser.add_argument('--path_save_dir', default='saved_models', help='base directory for saving results')
    parser.add_argument('--save_error_maps', type=str2bool, default=False, help='Save error map images')
    parser.add_argument('--overwrite', type=str2bool, default=True, help='overwrite previous results')
    
    opts = parser.parse_args()
    
    if not os.path.exists(opts.path_save_dir):
        os.makedirs(opts.path_save_dir)        
        
    #Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.path_save_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    
    if opts.predictions_file is not None:
        prediction_files = [opts.predictions_file]
        save_dirs = [opts.path_save_dir]
        
        train_or_tests = None
        structures = None
        
    else:
        #do a directory traversal
        prediction_files = glob.glob(opts.predictions_dir + '/*/*/predictions.csv')
        save_dirs = [path.replace(opts.predictions_dir, '') for path in prediction_files]
        save_dirs = [path.replace('predictions.csv', '') for path in save_dirs]
        save_dirs = [opts.path_save_dir + os.sep + path for path in save_dirs]
        
        #do some jenky parsing
        split_on_filesep = [path.split('/') for path in  prediction_files]
        train_or_tests = [split[-2] for split in split_on_filesep]
        structures = [split[-3] for split in split_on_filesep]
        
    stats_file = opts.path_save_dir + os.sep + 'stats.pkl' 
    if not opts.overwrite and os.path.exists(stats_file):
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

            path_pred_col = [column for column in df_preds.columns if 'path_prediction' in column][0]

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
    
    
    figure_save_path = opts.path_save_dir + os.sep + 'stats.png'

    print_stats_all(all_stats_list, figure_save_path)
    
    figure_save_path = opts.path_save_dir + os.sep + 'stats_v2.png'
    print_stats_all_v2(stats_per_im_list, figure_save_path)
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
if __name__ == '__main__':
    main()
