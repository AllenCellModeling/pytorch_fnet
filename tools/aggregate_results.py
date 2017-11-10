import argparse
import pandas as pd
import os
import pdb

def aggregate_results(path_results_dir, path_save_dir):
    """
    path_results_dir - path to directory of directories of results
    """
    def get_results_csv_path(path_search):
        for filename in [i.path for i in os.scandir(path_search) if i.is_file()]:
            if filename.lower().startswith('results') and filename.lower().endswith('.csv'):
                if filename.lower().endswith('_per.csv'):
                    continue
                return filename
        return None

    df_all = pd.DataFrame()
    for dirname in [i.path for i in os.scandir(path_results_dir) if i.is_dir()]:
        path_results = get_results_csv_path(dirname)
        if path_results is not None:
            df_entry = pd.read_csv(path_results)
            print('read csv:', path_results)
            df_all = pd.concat([df_all, df_entry], ignore_index=True)
    if path_save_dir is None:
        path_save_dir = path_results_dir
    path_results_all_csv = os.path.join(path_save_dir, 'results_all.csv')
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    df_all.to_csv(path_results_all_csv, index=False)
    print('wrote csv:', path_results_all_csv)

def main():
    """Look in results folders for fnet prediction results and combine into single csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_results_dir', required=True, help='path to results directory')
    parser.add_argument('--path_save_dir', help='path to save directory')
    opts = parser.parse_args()
    aggregate_results(opts.path_results_dir, opts.path_save_dir)
    
if __name__ == '__main__':
    main()
