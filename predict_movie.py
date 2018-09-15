import argparse
import fnet.data
import fnet.fnet_model
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
from fnet.utils import get_stats

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
        **opts.fds_kwargs
    )
    print(ds)
    return ds

def save_tiff_and_log(tag, ar, path_tiff_dir, entry, path_log_dir, batch_num):
    if not os.path.exists(path_tiff_dir):
        os.makedirs(path_tiff_dir)
    suffix = '' if batch_num is 0 else '_{:d}'.format(batch_num)
    path_tiff = os.path.join(path_tiff_dir, '{:s}.tiff'.format(tag + suffix))
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
    factor_yx = 0.37241/2  # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'fnet.transforms.Resizer((1/2, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--iter_mode', action='store_true', help='flag indicating whether to predict in iterative mode')
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
    parser.add_argument('--nn_kwargs', type=json.loads, default={}, help='kwargs to be passed to nn ctor')
    parser.add_argument('--fds_kwargs', type=json.loads, default={}, help='kwargs to be passed to FnetDataset')
    parser.add_argument('--nn_module', default='fnet_nn_3d', help='name of neural network module')
    
    opts = parser.parse_args()

    #if os.path.exists(opts.path_save_dir):
        #print('Output path already exists.')
        #return
    if opts.class_dataset == 'TiffDataset':
        if opts.propper_kwargs.get('action') == '-':
            opts.propper_kwargs['n_max_pixels'] = 6000000
    propper = fnet.transforms.Propper(**opts.propper_kwargs)
    print(propper)     
    dataset = get_dataset(opts, propper)
    dataset.val_mode = True
    #dataset.source_points = [8,2,1]
    
    models = {}
    for path_model_dir in opts.path_model_dir:
        model = fnet.fnet_model.Model(
            nn_module=opts.nn_module,
            gpu_ids=opts.gpu_ids,
            nn_kwargs=opts.nn_kwargs
        )
        # logger.info('Model instianted from: {:s}'.format(opts.nn_module))
        n_source_channels = opts.nn_kwargs["in_channels"]
        path_model = os.path.join(path_model_dir, 'model_{:s}.p'.format('' if not opts.iter_mode else '_'.join([str(i) for i in dataset.source_points])))
        model.load_state(path_model, gpu_ids=opts.gpu_ids)

        print(model)
        print(path_model)
        name_model = os.path.basename(path_model_dir)
        models[name_model] = model
        
    entries = []
    indices = range(len(dataset)) if opts.n_images < 0 else range(min(opts.n_images, len(dataset)))
    for idx in indices:
        if hasattr(dataset, "get_prediction_batch"):
            data = getattr(dataset, "get_prediction_batch")(idx)
        else:
            data = [torch.unsqueeze(d, 0) for d in dataset[idx]]  # make batch of size 1
        signal = data[0]
        target = data[1] if (len(data) > 1) else None
        n_batches = signal.size()[0]
        n_target_channels = target.size()[1]
        path_tiff_dir = os.path.join(opts.path_save_dir, '{:02d}'.format(idx))
        signalArr = signal.numpy()
        targetArr = target.numpy()
        sig_shape = [1,n_source_channels,16,240,240]
        
        for b in range(n_batches):
            entry = get_prediction_entry(dataset, idx)
            lag = b + dataset.time_slice_min
            if not opts.no_signal:
                save_tiff_and_log('signal', signalArr[b,0], path_tiff_dir, entry, opts.path_save_dir, lag)
            lag += dataset.n_source_points + dataset.n_offset - 1
            if not opts.no_target and target is not None:
                save_tiff_and_log('target', targetArr[b,0], path_tiff_dir, entry, opts.path_save_dir, lag)
            
            #this is necessary due to inexplicable torch failures with certain tensor sizes (might only affect old pytorch versions)
            sig_b = torch.zeros(sig_shape)
            sig_b[slice(None),slice(None),slice(None),slice(0,112),slice(0,176)] = signal[slice(b, b+1)]
            #sig_b = signal[slice(b,b+1)]
            for name_model in models:
                model = models[name_model]
                prediction = model.predict(sig_b)# if model is not None else None
                prediction = prediction.numpy()[:,:,:,0:112,0:176]
                pred_img = prediction[0, 0]
                tar_img = targetArr[b, 0]
                save_tiff_and_log('prediction', pred_img, path_tiff_dir, entry, opts.path_save_dir, lag)
                img_stats = get_stats(pred_img, tar_img)[2]
                entry['mse'] = img_stats['mse']
                entry['r'] = img_stats['r']
                if not opts.no_prediction_unpropped:
                    ar_pred_unpropped = propper.undo_last(prediction.numpy()[0, 0, ])
                    save_tiff_and_log('prediction_{:s}_unpropped'.format(name_model), ar_pred_unpropped, path_tiff_dir, entry, opts.path_save_dir, lag + chan)
            entries.append(entry)
        
    with open(os.path.join(opts.path_save_dir, 'predict_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)
    pd.DataFrame(entries).to_csv(os.path.join(opts.path_save_dir, 'predictions.csv'), index=False)
        

if __name__ == '__main__':
    main()
