import argparse
import fnet.data
import importlib
import json
import numpy as np
import os
import pandas as pd
import tifffile
import time
import torch
import warnings
import pdb

def set_warnings():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

def get_dataset(opts, propper):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    transform_signal.append(propper)
    transform_target.append(propper)
    ds = getattr(fnet.data, opts.class_dataset)(
        path_csv = opts.path_dataset_csv,
        transform_source = transform_signal,
        transform_target = transform_target,
    )
    print(ds)
    return ds

def save_tiff_and_log(tag, ar, path_tiff_dir, entry, path_log_dir):
    if not os.path.exists(path_tiff_dir):
        os.makedirs(path_tiff_dir)
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tiff'.format(tag))
    tifffile.imsave(path_tiff, ar)
    print('saved:', path_tiff)
    entry['path_' + tag] = os.path.relpath(path_tiff, path_log_dir)

def get_prediction_entry(dataset, index):
    info = dataset.get_information(index)
    # In the case where 'path_signal', 'path_target' keys exist in dataset information,
    # replace with 'path_signal_dataset', 'path_target_dataset' to avoid confusion with
    # predict.py's 'path_signal' and 'path_target'.
    if isinstance(info, dict):
        if 'path_signal' in info:
            info['path_signal_dataset'] = info.pop('path_signal')
        if 'path_target' in info:
            info['path_target_dataset'] = info.pop('path_target')
        return info
    if isinstance(info, str):
        return {'information': info}
    raise AttributeError
    
def main():
    # set_warnings()
    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--module_fnet_model', default='fnet_model', help='module with fnet_model')
    parser.add_argument('--n_images', type=int, default=16, help='max number of images to test')
    parser.add_argument('--no_prediction', action='store_true', help='set to not save prediction image')
    parser.add_argument('--no_prediction_unpropped', action='store_true', help='set to not save unpropped prediction image')
    parser.add_argument('--no_signal', action='store_true', help='set to not save signal image')
    parser.add_argument('--no_target', action='store_true', help='set to not save target image')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_model_dir', nargs='+', default=[None], help='path to model directory')
    parser.add_argument('--path_save_dir', help='path to output directory')
    parser.add_argument('--propper_kwargs', type=json.loads, default={}, help='path to output directory')
    parser.add_argument('--transform_signal', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset target')
    opts = parser.parse_args()

    if os.path.exists(opts.path_save_dir):
        print('Output path already exists.')
        return
    if opts.class_dataset == 'TiffDataset':
        if opts.propper_kwargs.get('action') == '-':
            opts.propper_kwargs['n_max_pixels'] = 6000000
    propper = fnet.transforms.Propper(**opts.propper_kwargs)
    print(propper)
    model = None
    dataset = get_dataset(opts, propper)
    entries = []
    indices = range(len(dataset)) if opts.n_images < 0 else range(min(opts.n_images, len(dataset)))
    for idx in indices:
        entry = get_prediction_entry(dataset, idx)
        data = [torch.unsqueeze(d, 0) for d in dataset[idx]]  # make batch of size 1
        signal = data[0]
        target = data[1] if (len(data) > 1) else None
        path_tiff_dir = os.path.join(opts.path_save_dir, '{:02d}'.format(idx))
        if not opts.no_signal:
            save_tiff_and_log('signal', signal.numpy()[0, ], path_tiff_dir, entry, opts.path_save_dir)
        if not opts.no_target and target is not None:
            save_tiff_and_log('target', target.numpy()[0, ], path_tiff_dir, entry, opts.path_save_dir)

        for path_model_dir in opts.path_model_dir:
            if (path_model_dir is not None) and (model is None or len(opts.path_model_dir) > 1):
                model = fnet.load_model(path_model_dir, opts.gpu_ids, module=opts.module_fnet_model)
                print(model)
                name_model = os.path.basename(path_model_dir)
            prediction = model.predict(signal) if model is not None else None
            if not opts.no_prediction and prediction is not None:
                save_tiff_and_log('prediction_{:s}'.format(name_model), prediction.numpy()[0, ], path_tiff_dir, entry, opts.path_save_dir)
            if not opts.no_prediction_unpropped:
                ar_pred_unpropped = propper.undo_last(prediction.numpy()[0, 0, ])
                save_tiff_and_log('prediction_{:s}_unpropped'.format(name_model), ar_pred_unpropped, path_tiff_dir, entry, opts.path_save_dir)
        entries.append(entry)
        
    with open(os.path.join(opts.path_save_dir, 'predict_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)
    pd.DataFrame(entries).to_csv(os.path.join(opts.path_save_dir, 'predictions.csv'), index=False)
        

if __name__ == '__main__':
    main()
