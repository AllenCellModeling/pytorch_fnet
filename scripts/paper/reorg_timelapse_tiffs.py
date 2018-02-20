import argparse
import os
import pandas as pd
import shutil
import pdb

"""Script to convert 3d timelapse predictions to match format of 2d slices used to make gifs."""

MAP_NAMING = {
    'signal': 'bf',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_input_csv', required=True, help='path to input csv')
    parser.add_argument('-o', '--path_output_dir', required=True, help='destination directory')
    args = parser.parse_args()

    df = pd.read_csv(args.path_input_csv)
    cols = [col for col in df.columns if 'path_prediction_' in col]
    cols.append('path_signal')
    print(cols)

    # each column becomes its own directory without path_output_dir
    paths_src = list()
    paths_dst = list()
    for col in cols:
        print('processing:' ,col)
        dirname_dst_partial = col.split('path_')[-1].split('prediction_')[-1]
        dirname_dst = os.path.join(
            args.path_output_dir,
            MAP_NAMING.get(dirname_dst_partial, dirname_dst_partial),
        )
        if not os.path.exists(dirname_dst):
            os.makedirs(dirname_dst)
        for src in df[col]:
            path_src = os.path.join(os.path.dirname(args.path_input_csv), src)
            basename_dst = '_'.join(os.path.split(src))
            for key, value in MAP_NAMING.items():
                basename_dst = basename_dst.replace(key, value)
            path_dst = os.path.join(
                dirname_dst,
                basename_dst,
            )
            shutil.copy(path_src, path_dst)
            paths_src.append(os.path.abspath(path_src))
            paths_dst.append(os.path.relpath(path_dst, args.path_output_dir))
    df_manifest = pd.DataFrame({'path_src': paths_src, 'path_dst': paths_dst})[
        ['path_src', 'path_dst']]
    path_manifest = os.path.join(
        args.path_output_dir,
        os.path.basename(__file__).split('.')[0] + '.csv',
    )
    df_manifest.to_csv(path_manifest, index=False)
    print('saved:', path_manifest)

if __name__ == '__main__':
    main()

