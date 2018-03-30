import argparse
import os
import pandas as pd
import time
import subprocess
import pdb

PATHS_DEFAULT = [
    'data/csvs/tester.csv',
    'data/csvs/alpha_tubulin.csv',
    'data/csvs/beta_actin.csv',
    'data/csvs/desmoplakin.csv',
    'data/csvs/dic_lamin_b1.csv',
    'data/csvs/fibrillarin.csv',
    'data/csvs/lamin_b1.csv',
    'data/csvs/membrane_caax_63x.csv',
    'data/csvs/myosin_iib.csv',
    'data/csvs/sec61_beta.csv',
    'data/csvs/st6gal1.csv',
    # 'data/csvs/timelapse_wt2_s2.csv',
    'data/csvs/tom20.csv',
    'data/csvs/zo1.csv',
]

def create_czi_symlinks(path_csv, path_save_dir):
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    df = pd.read_csv(path_csv)
    for path_czi in df['path_czi']:
        path_dst = os.path.join(
            path_save_dir,
            os.path.basename(path_czi),
        )
        if os.path.exists(path_dst):
            os.unlink(path_dst)
        os.symlink(path_czi, path_dst)
        print('created:', path_dst)

def zip_data(path_src, path_dst):
    if os.path.exists(path_dst):
        print(path_dst, 'already exists. skipping....')
        return
    dirname = os.path.dirname(path_src)
    basename = os.path.basename(path_src)
    cmd = 'cd {:s}; tar -czhvf {:s} {:s}'.format(dirname, path_dst, basename)
    print('executing:', cmd)
    subprocess.run(cmd, shell=True)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--paths_input_csv', nargs='+', default=PATHS_DEFAULT, help='path(s) to input csv')
    parser.add_argument('-o', '--path_save_dir', help='path(s) to input csv')
    args = parser.parse_args()

    path_save_dir = args.path_save_dir if args.path_save_dir is not None else os.getcwd()
    print(path_save_dir)
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    path_root = os.path.join(os.path.dirname(__file__), '../../..')
    time_genesis = time.time()
    for p in args.paths_input_csv:
        time_start = time.time()
        print('processing:', p)
        name = os.path.basename(p).split('.csv')[0]
        path_csv = os.path.join(path_root, p)
        path_sym_dir = os.path.join(path_root, 'data', name)
        # create_czi_symlinks(path_csv, path_sym_dir)
        path_tar = os.path.join(path_save_dir, '{:s}.tar.gz'.format(name))
        zip_data(path_sym_dir, path_tar)
        print('elapsed time: {:.1f}'.format(time.time() - time_start))
    print('total elapsed time: {:.1f}'.format(time.time() - time_genesis))


if __name__ == '__main__':
    main()

