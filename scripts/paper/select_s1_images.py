import argparse
import numpy as np
import os
import pandas as pd
import json
import tifffile
import shutil
import pdb

def to_uint8(ar, val_range=None):
    ar = ar.copy()
    if val_range is None:
        raise NotImplementedError
        val_min, val_max = np.percentile(ar, 0.1), np.percentile(ar, 99.9)
    else:
        val_min, val_max = val_range
    print(val_min, val_max)
    ar[ar <= val_min] = val_min
    ar[ar >= val_max] = val_max
    ar = (ar - val_min)/(val_max - val_min)*256.0
    ar[ar >= 256.0] = 255
    return ar.astype(np.uint8)

def finder_middle(ar):
    if ar.ndim == 4:
        ar = ar[0, ]
    return ar.shape[0]//2

def finder_max(ar):
    if ar.ndim == 4:
        ar = ar[0, ]
    return np.argmax(np.sum(ar, axis=(1, 2)))

def finder_z52(ar):
    return 52

MAP_SLICE_FINDER = {
    'alpha_tubulin': finder_middle,
    'beta_actin': finder_middle,
    'dic_lamin_b1': finder_middle,
    'dic_membrane': finder_middle,
    'lamin_b1': finder_middle,
    'membrane': finder_middle,
    'sec61_beta': finder_middle,
    'tom20': finder_middle,
    'myosin_iib': finder_z52,
}

def select_files(paths_inputs, path_output_dir, seed, overwrite=False):
    path_manifest = os.path.join(path_output_dir, os.path.basename(__file__).split('.')[0] + '.csv')
    if not overwrite and os.path.exists(path_manifest):
        print('using existing:', path_manifest)
        return path_manifest
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    rng = np.random.RandomState(seed)
    entries = list()
    for path_input in paths_inputs:
        if not os.path.exists(path_input):
            print('*** skipping (does not exist):', path_input)
            continue
        print('processing:', path_input)
        df = pd.read_csv(path_input)
        idx_random = rng.randint(df.shape[0])
        print('=> using index', idx_random)
        cols_predictions = [c for c in df.columns if 'path_prediction_' in c]
        assert len(cols_predictions) == 1
        name_model = cols_predictions[0].split('path_prediction_')[-1]
        dirname_src = os.path.dirname(os.path.abspath(path_input))
        path_src_target = os.path.join(dirname_src, df.loc[df.index[idx_random], 'path_target'])
        path_src_prediction = os.path.join(dirname_src, df.loc[df.index[idx_random], cols_predictions[0]])
        path_dst_target = os.path.join(path_output_dir, name_model + '_target.tiff')
        path_dst_prediction = os.path.join(path_output_dir, name_model + '_prediction.tiff')

        shutil.copy(path_src_target, path_dst_target)
        shutil.copy(path_src_prediction, path_dst_prediction)
        entry = {
            'model': name_model,
            'path_src_target': path_src_target,
            'path_src_prediction': path_src_prediction,
            'path_dst_target': os.path.relpath(path_dst_target, path_output_dir),
            'path_dst_prediction': os.path.relpath(path_dst_prediction, path_output_dir), 
        }
        entries.append(entry)

    df_manifest = pd.DataFrame(entries, columns=['model', 'path_src_target', 'path_src_prediction', 'path_dst_target', 'path_dst_prediction'])
    df_manifest.to_csv(path_manifest, index=False)
    print('saved:', path_manifest)
    return path_manifest

def select_slices(path_csv, path_output_dir):
    df = pd.read_csv(path_csv)
    dirname_csv = os.path.dirname(path_csv)
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    entries = list()
    for idx, row in df.iterrows():
        path_target = os.path.join(dirname_csv, row['path_dst_target'])
        path_prediction = os.path.join(dirname_csv, row['path_dst_prediction'])
        ar_t = tifffile.imread(path_target)
        ar_p = tifffile.imread(path_prediction)
        fn_finder = MAP_SLICE_FINDER.get(row['model'], finder_max)
        idx_z = fn_finder(ar_t)
        path_img_target = os.path.join(
            path_output_dir,
            os.path.basename(path_target).split('.')[0] + '_z{:02d}.tiff'.format(idx_z),
        )
        path_img_prediction = os.path.join(
            path_output_dir,
            os.path.basename(path_prediction).split('.')[0] + '_z{:02d}.tiff'.format(idx_z),
        )
        val_range = np.percentile(ar_t[0, idx_z, ], (.1, 99.9))
        img_t = to_uint8(ar_t[0, idx_z, ] if ar_t.ndim == 4 else ar_t[idx_z, ], val_range=val_range)
        img_p = to_uint8(ar_p[0, idx_z, ] if ar_p.ndim == 4 else ar_p[idx_z, ], val_range=val_range)
        tifffile.imsave(path_img_target, img_t)
        print('saved:', path_img_target)
        tifffile.imsave(path_img_prediction, img_p)
        print('saved:', path_img_prediction)
        entry = {
            'model': row['model'],
            'slice_finder': fn_finder.__name__,
            'z_slice': idx_z,
            'val_range': val_range,
            'path_img_target': os.path.relpath(path_img_target, path_output_dir),
            'path_img_prediction': os.path.relpath(path_img_target, path_output_dir),
        }
        entries.append(entry)
    path_output_csv = os.path.join(path_output_dir, 'slices.csv')
    pd.DataFrame(entries).to_csv(path_output_csv, index=False)
    print('saved:', path_output_csv)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--paths_inputs', nargs='+', required=True, help='path to input predictions.csv(s)')
    parser.add_argument('-o', '--path_output_dir', required=True, help='destination directory')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--overwrite', action='store_true', help='set to existing overwrite files')
    args = parser.parse_args()
    
    path_manifest = select_files(args.paths_inputs, os.path.join(args.path_output_dir, '3d'), args.seed, overwrite=args.overwrite)
    select_slices(path_manifest, os.path.join(args.path_output_dir, '2d'))

    dict_out = vars(args)
    dict_out['__file__'] = __file__
    for k, v in dict_out.items():
        if k in ['paths_inputs', 'path_output_dir', '__file__']:
            if isinstance(v, str):
                dict_out[k] = os.path.abspath(v)
            elif isinstance(v, list):
                dict_out[k] = [os.path.abspath(p) for p in v]
    path_json = os.path.join(
        args.path_output_dir,
        os.path.basename(__file__).split('.')[0] + '.json',
    )
    with open(path_json, 'w') as fo:
        json.dump(dict_out, fp=fo, indent=4)
        print('saved:', path_json)

        
if __name__ == '__main__':
    main()

