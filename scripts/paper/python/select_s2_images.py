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

def select_files(paths_inputs, path_output_dir, seed=None, overwrite=False, specify=None, crop=None, include_signal=False):
    if specify is not None:
        assert len(specify) == len(paths_inputs)
    path_manifest = os.path.join(path_output_dir, os.path.basename(__file__).split('.')[0] + '.csv')
    if not overwrite and os.path.exists(path_manifest):
        print('using existing:', path_manifest)
        return path_manifest
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    rng = np.random.RandomState(seed)
    entries = list()
    for idx_path, path_input in enumerate(paths_inputs):
        if not os.path.exists(path_input):
            print('*** skipping (does not exist):', path_input)
            continue
        print('processing:', path_input)
        df = pd.read_csv(path_input)
        if specify is not None:
            idx_select = specify[idx_path]
        else:
            idx_select = rng.randint(df.shape[0])
        print('=> using index', idx_select)
        cols_predictions = [c for c in df.columns if 'path_prediction_' in c]
        assert len(cols_predictions) == 1
        name_model = cols_predictions[0].split('path_prediction_')[-1]
        dirname_src = os.path.dirname(os.path.abspath(path_input))
        path_src_target = os.path.join(dirname_src, df.loc[df.index[idx_select], 'path_target'])
        path_src_prediction = os.path.join(dirname_src, df.loc[df.index[idx_select], cols_predictions[0]])
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
        if include_signal and 'path_signal' in df.columns:
            path_src_signal = os.path.join(dirname_src, df.loc[df.index[idx_select], 'path_signal'])
            path_dst_signal = os.path.join(path_output_dir, name_model + '_signal.tiff')
            shutil.copy(path_src_signal, path_dst_signal)
            entry['path_src_signal'] = path_src_signal
            entry['path_dst_signal'] = os.path.relpath(path_dst_signal, path_output_dir)
        entries.append(entry)
    df_manifest = pd.DataFrame(entries)
    df_manifest.to_csv(path_manifest, index=False)
    print('saved:', path_manifest)
    return path_manifest

def find_key(keys, x):
    # find key in keys where key is a string within x
    for k in keys:
        if k in x:
            return k
    return None

def calc_slices_crop(shape, crop):
    # calculate slices to crop from center of image of specified shape
    slices = list()
    for d in range(len(shape)):
        start = (shape[d] - crop[d])//2
        slices.append(slice(start, start + crop[d]))
    return slices

def select_slices(path_csv, path_output_dir, indices=None, crop=None, include_signal=False):
    df = pd.read_csv(path_csv)
    if indices is not None:
        assert len(indices) == df.shape[0]
    if crop is not None:
        assert len(crop) == 2
    dirname_csv = os.path.dirname(path_csv)
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    entries = list()
    for idx, row in df.iterrows():
        path_target = os.path.join(dirname_csv, row['path_dst_target'])
        path_prediction = os.path.join(dirname_csv, row['path_dst_prediction'])
        ar_t = tifffile.imread(path_target)
        ar_p = tifffile.imread(path_prediction)
        ar_t = ar_t[0, ] if ar_t.ndim == 4 else ar_t
        ar_p = ar_p[0, ] if ar_p.ndim == 4 else ar_p
        if indices is not None and indices[idx].isdigit():
            idx_z = int(indices[idx])
        else:
            key_model = find_key(MAP_SLICE_FINDER.keys(), row['model'])
            fn_finder = MAP_SLICE_FINDER[key_model] if key_model is not None else finder_max
            idx_z = fn_finder(ar_t)
        path_img_target = os.path.join(
            path_output_dir,
            os.path.basename(path_target).split('.')[0] + '_z{:02d}.tiff'.format(idx_z),
        )
        path_img_prediction = os.path.join(
            path_output_dir,
            os.path.basename(path_prediction).split('.')[0] + '_z{:02d}.tiff'.format(idx_z),
        )
        val_range = np.percentile(ar_t[idx_z, ], (.1, 99.9))
        img_t = to_uint8(ar_t[idx_z, ], val_range=val_range)
        img_p = to_uint8(ar_p[idx_z, ], val_range=val_range)
        if crop is not None and all(img_t.shape[i] > crop[i] for i in range(img_t.ndim)):
            slices = calc_slices_crop(img_t.shape, crop)
            shape_old = img_t.shape
            img_t = img_t[slices]
            img_p = img_p[slices]
            print('doing crop: {} => {}'.format(shape_old, img_t.shape))
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
        if include_signal:
            path_signal = os.path.join(dirname_csv, row['path_dst_signal'])
            path_img_signal = os.path.join(
                path_output_dir,
                os.path.basename(path_signal).split('.')[0] + '_z{:02d}.tiff'.format(idx_z),
            )
            ar_s = tifffile.imread(path_signal)
            ar_s = ar_s[0, ] if ar_s.ndim == 4 else ar_s
            val_range_s = np.percentile(ar_s[idx_z, ], (.1, 99.9))
            img_s = to_uint8(ar_s[idx_z, ], val_range=val_range_s)
            tifffile.imsave(path_img_signal, img_s)
            print('saved:', path_img_signal)
            
        entries.append(entry)
    path_output_csv = os.path.join(path_output_dir, 'slices.csv')
    pd.DataFrame(entries).to_csv(path_output_csv, index=False)
    print('saved:', path_output_csv)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--paths_inputs', nargs='+', required=True, help='path to input predictions.csv(s)')
    parser.add_argument('-o', '--path_save_dir', required=True, help='destination directory')
    parser.add_argument('--include_signal', action='store_true', help='set to include input (transmitted light) image')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--specify', nargs='+', type=int, help='specify images from each predictions.csv file')
    parser.add_argument('--z_slices', nargs='+', help='specify z-slice for each 3d volume')
    parser.add_argument('--crop', nargs='+', type=int, help='specify crop size')
    parser.add_argument('--overwrite', action='store_true', help='set to existing overwrite files')
    args = parser.parse_args()

    path_manifest = select_files(
        args.paths_inputs, os.path.join(args.path_save_dir, '3d'),
        args.seed, overwrite=args.overwrite, specify=args.specify,
        include_signal=args.include_signal,
    )
    select_slices(
        path_manifest, os.path.join(args.path_save_dir, '2d'), args.z_slices, crop=args.crop,
        include_signal=args.include_signal,          
    )

    dict_out = vars(args)
    dict_out['__file__'] = __file__
    for k, v in dict_out.items():
        if k in ['paths_inputs', 'path_save_dir', '__file__']:
            if isinstance(v, str):
                dict_out[k] = os.path.abspath(v)
            elif isinstance(v, list):
                dict_out[k] = [os.path.abspath(p) for p in v]
    path_json = os.path.join(
        args.path_save_dir,
        os.path.basename(__file__).split('.')[0] + '.json',
    )
    with open(path_json, 'w') as fo:
        json.dump(dict_out, fp=fo, indent=4)
        print('saved:', path_json)

        
if __name__ == '__main__':
    main()

