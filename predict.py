import argparse
import fnet.data
import importlib
import pandas as pd
import torch
import tifffile
import time
import os
import warnings
import json
import pdb

def set_warnings():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

def get_dataset(opts, cropper):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    transform_signal.append(cropper)
    transform_target.append(cropper)
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

def main():
    # set_warnings()
    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--dont_save_signal', action='store_true', help='set to not save signal image')
    parser.add_argument('--dont_save_target', action='store_true', help='set to not save target image')
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--n_images', type=int, default=16, help='max number of images to test')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_model_dir', help='path to model directory')
    parser.add_argument('--path_save_dir', help='path to output directory')
    parser.add_argument('--transform_signal', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset target')
    opts = parser.parse_args()

    if opts.class_dataset == 'CziDataset':
        cropper = fnet.transforms.Cropper(('/16', '/16', '/16'), offsets=('mid', 'mid', 'mid'))
    else:  # opts.class_dataset == 'TiffDataset'
        cropper = fnet.transforms.Cropper(('/16', '/16'), offsets=('mid', 'mid'),
                                          n_max_pixels=6000000)
        
    model = fnet.load_model_from_dir(opts.path_model_dir, opts.gpu_ids)
    print(model)
    dataset = get_dataset(opts, cropper)
    entries = []
    count = 0
    for i, data_pre in enumerate(dataset):
        info_data = dataset.get_information(i)
        entry = {'information': info_data} if isinstance(info_data, str) else info_data
        data = [torch.unsqueeze(d, 0) for d in data_pre]  # make batch of size 1
        signal = data[0]
        target = data[1] if (len(data) > 1) else None
        prediction = model.predict(signal)
        path_tiff_dir = os.path.join(opts.path_save_dir, '{:02d}'.format(i))
        if not opts.dont_save_signal:
            save_tiff_and_log('signal_transformed', signal.numpy()[0, ], path_tiff_dir, entry, opts.path_save_dir)
        if not opts.dont_save_target and target is not None:
            save_tiff_and_log('target_transformed', target.numpy()[0, ], path_tiff_dir, entry, opts.path_save_dir)
        save_tiff_and_log('prediction_propped', prediction.numpy()[0, ], path_tiff_dir, entry, opts.path_save_dir)
        
        ar_pred_unpropped = cropper.unprop(prediction.numpy()[0, 0, ])
        save_tiff_and_log('prediction', ar_pred_unpropped, path_tiff_dir, entry, opts.path_save_dir)
        
        entries.append(entry)
        count += 1
        if count >= opts.n_images:
            break
    with open(os.path.join(opts.path_save_dir, 'predict_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)
    pd.DataFrame(entries).to_csv(os.path.join(opts.path_save_dir, 'predictions.csv'))
        

if __name__ == '__main__':
    main()
