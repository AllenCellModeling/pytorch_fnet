import argparse
import os
import pandas as pd
import time
import subprocess
import pdb

dirname_script = os.path.dirname(__file__)
# no need to include a "DNA" folder because the files should come from other folders
# os.path.join(dirname_script, 'source_csvs/dna.csv'),  
PATHS_DEFAULT = [
    os.path.join(dirname_script, 'source_csvs/alpha_tubulin.csv'),
    os.path.join(dirname_script, 'source_csvs/beta_actin.csv'),
    os.path.join(dirname_script, 'source_csvs/desmoplakin.csv'),
    os.path.join(dirname_script, 'source_csvs/dic_lamin_b1.csv'),
    os.path.join(dirname_script, 'source_csvs/fibrillarin.csv'),
    os.path.join(dirname_script, 'source_csvs/lamin_b1.csv'),
    os.path.join(dirname_script, 'source_csvs/membrane_caax_63x.csv'),
    os.path.join(dirname_script, 'source_csvs/myosin_iib.csv'),
    os.path.join(dirname_script, 'source_csvs/sec61_beta.csv'),
    os.path.join(dirname_script, 'source_csvs/st6gal1.csv'),
    os.path.join(dirname_script, 'source_csvs/timelapse_wt2_s2.csv'),
    os.path.join(dirname_script, 'source_csvs/tom20.csv'),
    os.path.join(dirname_script, 'source_csvs/zo1.csv'),
]

def get_map_file_to_link(path_root):
    paths_dirs = [i.path for i in os.scandir(os.path.join(path_root, 'data')) if i.is_dir()]
    mapping = dict()
    for path_dir in paths_dirs:
        for path_f in os.scandir(path_dir):
            if (not path_f.is_symlink() or
                not any(path_f.path.lower().endswith(i) for i in ['.czi', '.tif', '.tiff']) ):
                continue
            path_relative = os.path.relpath(path_f, path_root)
            basename = os.path.basename(path_relative)
            mapping[basename] = path_relative
    return mapping

def create_rel_csv(path_src_csv, path_dst_csv, map_file_to_link):
    df = pd.read_csv(path_src_csv)
    for col in ['path_czi', 'path_signal', 'path_target']:
        if col not in df:
            continue
        df[col] = df[col].apply(lambda x: map_file_to_link[os.path.basename(x)] if isinstance(x, str) else None)
    df.to_csv(path_dst_csv, index=False)

def create_czi_symlinks(path_csv, path_save_dir):
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    df = pd.read_csv(path_csv)
    if 'path_czi' in df.columns:
        cols = ['path_czi']
    else:  # assume tiff-based dataset
        cols = ['path_signal', 'path_target']
    for idx, row in df.iterrows():
        for c in cols:
            path_file = row[c]
            if not isinstance(path_file, str):  # path_target might be NA
                continue
            path_dst = os.path.join(
                path_save_dir,
                os.path.basename(path_file),
            )
            if os.path.exists(path_dst):
                os.unlink(path_dst)
            os.symlink(path_file, path_dst)
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
    parser.add_argument('--create_rel_csvs', action='store_true', help='set to create csv of czis using relative paths')
    parser.add_argument('--zip_data', action='store_true', help='set to tarball data folders')
    args = parser.parse_args()

    path_save_dir = args.path_save_dir if args.path_save_dir is not None else os.getcwd()
    print(path_save_dir)
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    path_root = os.path.join(os.path.dirname(__file__), '../../..')
    time_genesis = time.time()
    if args.create_rel_csvs:
        map_file_to_link = get_map_file_to_link(path_root)
    for path_csv in args.paths_input_csv:
        time_start = time.time()
        print('processing:', path_csv)
        name = os.path.basename(path_csv).split('.csv')[0]
        path_sym_dir = os.path.join(path_root, 'data', name)
        create_czi_symlinks(path_csv, path_sym_dir)
        if args.create_rel_csvs:
            path_dst_csv = os.path.join(path_root, 'data', 'csvs', os.path.basename(path_csv))
            create_rel_csv(path_csv, path_dst_csv, map_file_to_link)
        if args.zip_data:
            path_tar = os.path.join(path_save_dir, '{:s}.tar.gz'.format(name))
            zip_data(path_sym_dir, path_tar)
        print('elapsed time: {:.1f}'.format(time.time() - time_start))
    print('total elapsed time: {:.1f}'.format(time.time() - time_genesis))


if __name__ == '__main__':
    main()

