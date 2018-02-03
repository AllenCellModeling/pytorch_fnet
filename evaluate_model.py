import argparse

from fnet.utils.figures import eval_images

import logging
import os
import pandas as pd
import sys
import warnings

import numpy as np
import glob


import pdb

    
def main():
    
    parser = argparse.ArgumentParser()
    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser.add_argument('--predictions_file', default=None, help='two column .csv of paths to path_prediction and path_target')
    parser.add_argument('--predictions_dir', default=None, help='project directory. Mutually exclusive of predictions_dir')
    parser.add_argument('--path_save_dir', default='saved_models', help='base directory for saving results')
    parser.add_argument('--save_error_maps', type=bool, default=False, help='Save error map images')
    
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
    else:
        #do a directory traversal
        prediction_files = glob.glob(opts.predictions_dir + '/*/*/predictions.csv')
        save_dirs = [path.replace(opts.predictions_dir, '') for path in prediction_files]
        save_dirs = [path.replace('predictions.csv', '') for path in save_dirs]
        
    for prediction_file, save_dir in zip(prediction_files, save_dirs):
    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        pred_dir, _ = os.path.split(prediction_file)

        df_preds = pd.read_csv(prediction_file)

        #prepend the directory to the paths in the dataframe
        path_columns = [column for column in df_preds.columns if 'path' in column]
        for column in path_columns:
            # pdb.set_trace()
            
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
    

if __name__ == '__main__':
    main()
