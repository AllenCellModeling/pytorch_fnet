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

def get_dataloader(opts):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    if opts.class_dataset == 'CziDataset':
        cropper = fnet.transforms.Cropper(('/16', '/16', '/16'), offsets=('mid', 'mid', 'mid'))
    elif opts.class_dataset == 'TiffDataset':
        cropper = fnet.transforms.Cropper(('/16', '/16'), offsets=('mid', 'mid'),
                                          n_max_pixels=6000000
        )
    else:
        raise NotImplementedError
    transform_signal.append(cropper)
    transform_target.append(cropper)
    ds = getattr(fnet.data, opts.class_dataset)(
        path_csv = opts.path_dataset_csv,
        transform_source = transform_signal,
        transform_target = transform_target,
    )
    print(ds)
    return torch.utils.data.DataLoader(
        ds,
        batch_size = 1,
    )

def main():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--n_images', type=int, default=16, help='max number of images to test')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_model_dir', help='path to model directory')
    parser.add_argument('--path_save_dir', help='path to output directory')
    parser.add_argument('--transform_signal', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset target')
    opts = parser.parse_args()

    model = fnet.load_model_from_dir(opts.path_model_dir, opts.gpu_ids)
    print(model)
    dataloader = get_dataloader(opts)
    entries = []
    for i, (signal, target) in enumerate(dataloader):
        prediction = model.predict(signal)
        path_tiff_dir = os.path.join(opts.path_save_dir, '{:02d}'.format(i))
        if not os.path.exists(path_tiff_dir):
            os.makedirs(path_tiff_dir)
        path_tiff_s = os.path.join(path_tiff_dir, 'signal.tiff')
        path_tiff_t = os.path.join(path_tiff_dir, 'target.tiff')
        path_tiff_p = os.path.join(path_tiff_dir, 'prediction.tiff')
        tifffile.imsave(path_tiff_s, signal.numpy()[0, ])
        print('saved:', path_tiff_s)
        tifffile.imsave(path_tiff_t, target.numpy()[0, ])
        print('saved:', path_tiff_t)
        tifffile.imsave(path_tiff_p, prediction.numpy()[0, ])
        print('saved:', path_tiff_p)
        entries.append({
            'path_signal': os.path.relpath(path_tiff_s, opts.path_save_dir),
            'path_target': os.path.relpath(path_tiff_t, opts.path_save_dir),
            'path_prediction': os.path.relpath(path_tiff_p, opts.path_save_dir),
        })
        if i >= opts.n_images:
            break
    with open(os.path.join(opts.path_save_dir, 'predict_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)
    pd.DataFrame(entries).to_csv(os.path.join(opts.path_save_dir, 'predictions.csv'))
        

if __name__ == '__main__':
    main()
