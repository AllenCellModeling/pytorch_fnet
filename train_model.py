import os
import argparse
import importlib
import fnet
import fnet.data
import fnet.data.transforms
import pandas as pd
import numpy as np
import torch
import pdb
import time
import logging
import sys
import shutil
import json
import warnings

def train_model(**kwargs):
    start = time.time()
    model = kwargs.get('model')
    data_provider = kwargs.get('data_provider')
    logger = kwargs.get('logger')
    path_run_dir = kwargs.get('path_run_dir')
    n_iter = kwargs.get('n_iter')
    iter_checkpoint = kwargs.get('iter_checkpoint', n_iter)
    data_provider_nonchunk = kwargs.get('data_provider_nonchunk')

    assert model is not None
    assert data_provider is not None
    assert path_run_dir is not None
    assert n_iter is not None
    
    print_fn = logger.info if logger is not None else print

    path_existing_model = os.path.join(path_run_dir, 'model.p')
    path_existing_run_state = os.path.join(path_run_dir, 'run_state.p')
    if os.path.exists(path_existing_model) and os.path.exists(path_existing_run_state):
        model.load_state(path_existing_model)
        loss_log = fnet.load_run_state(path_existing_run_state)
    else:
        loss_log = fnet.SimpleLogger(
            ('num_iter', 'loss', 'sources'),
            'num_iter: {:4d} | loss: {:.4f} | sources: {:s}',
        )
    print_fn(model)
    
    df_checkpoints = pd.DataFrame()
    start_iter = model.count_iter
    for i in range(start_iter, n_iter):
        x, y = data_provider.get_batch()
        loss = model.do_train_iter(x, y)
        str_out = loss_log.add((
            i + 1,
            loss,
            data_provider.last_sources
        ))
        print_fn(str_out)
        if ((i + 1) % iter_checkpoint == 0) or ((i + 1) == n_iter):
            path_checkpoint_dir = os.path.join(path_run_dir, 'output_{:05d}'.format(i + 1))
            loss_log.save_csv(os.path.join(path_run_dir, 'loss_log.csv'))
            model.save_state(os.path.join(path_run_dir, 'model.p'))
            fnet.save_run_state(os.path.join(path_run_dir, 'run_state.p'), loss_log)
            if data_provider_nonchunk is not None:
                kwargs_checkpoint = dict(
                    n_images = 4,
                    save_images = True,
                    path_save = path_checkpoint_dir,
                )
                data_provider_nonchunk.use_train_set()
                losses_checkpoint = fnet.test_model(model, data_provider_nonchunk, **kwargs_checkpoint)
                data_provider_nonchunk.use_test_set()
                losses_checkpoint.update(fnet.test_model(model, data_provider_nonchunk, **kwargs_checkpoint))
                losses_checkpoint['num_iter'] = i
                df_checkpoints = pd.concat([df_checkpoints, pd.DataFrame([losses_checkpoint])], ignore_index=True)
                df_checkpoints.to_csv(os.path.join(path_checkpoint_dir, 'losses_checkpoint.csv'), index=False)
    t_elapsed = time.time() - start
    print_fn('total training time: {:.1f} s'.format(t_elapsed))
    print(df_checkpoints)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
    parser.add_argument('--buffer_size', type=int, default=5, help='number of images to cache in memory')
    parser.add_argument('--path_train_csv', help='path to training set csv')
    parser.add_argument('--path_test_csv', help='path to test set csv')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
    parser.add_argument('--iter_checkpoint', type=int, default=500, help='iterations between saving log/model checkpoints')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model_module', default='fnet_model', help='name of the model module')
    parser.add_argument('--n_iter', type=int, default=500, help='number of training iterations')
    parser.add_argument('--scale_z', type=float, default=0.3, help='desired um/px scale for z dimension')
    parser.add_argument('--scale_xy', type=float, default=0.3, help='desired um/px scale for x, y dimensions')
    parser.add_argument('--transforms_signal', nargs='+', default=['fnet.data.sub_mean_norm'], help='transform to be applied to signal images')
    parser.add_argument('--transforms_target', nargs='+', default=['fnet.data.sub_mean_norm'], help='transform to be applied to target images')
    parser.add_argument('--nn_module', default='ttf_v8_nn', help='name of neural network module')
    parser.add_argument('--replace_interval', type=int, default=-1, help='iterations between replacements of images in cache')
    parser.add_argument('--path_run_dir', default='saved_models', help='base directory for saved models')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    opts = parser.parse_args()
    model_module = importlib.import_module('model_modules.' + opts.model_module)
    
    if not os.path.exists(opts.path_run_dir):
        os.makedirs(opts.path_run_dir)
    
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.path_run_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    warnings.showwarning = lambda *args, **kwargs : logger.warning(warnings.formatwarning(*args, **kwargs))

    main_gpu_id = opts.gpu_ids if isinstance(opts.gpu_ids, int) else opts.gpu_ids[0]
    torch.cuda.set_device(main_gpu_id)
    logger.info('main GPU ID: {:d}'.format(torch.cuda.current_device()))

    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    model = model_module.Model(
        nn_module=opts.nn_module,
        lr=opts.lr,
        gpu_ids=opts.gpu_ids,
    )

    path_ds = os.path.join(opts.path_run_dir, 'ds.json')
    fnet.data.save_dataset_as_json(
        path_train_csv = opts.path_train_csv,
        path_test_csv = opts.path_test_csv,
        scale_z = opts.scale_z,
        scale_xy = opts.scale_xy,
        transforms_signal = opts.transforms_signal,
        transforms_target = opts.transforms_target,
        path_save = path_ds,
    )
    dataset = fnet.data.load_dataset_from_json(path_load = path_ds)
    logger.info(dataset)
    
    shutil.copyfile(opts.path_train_csv, os.path.join(opts.path_run_dir, os.path.basename(opts.path_train_csv)))
    shutil.copyfile(opts.path_test_csv, os.path.join(opts.path_run_dir, os.path.basename(opts.path_test_csv)))
    
    data_provider = fnet.data.ChunkDataProvider(
        dataset,
        buffer_size=opts.buffer_size,
        batch_size=opts.batch_size,
        replace_interval=opts.replace_interval,
    )
    
    dims_cropped = (32, '/16', '/16')
    cropper = fnet.data.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
    transforms_nonchunk = (cropper, cropper)
    data_provider_nonchunk = fnet.data.TestImgDataProvider(
        dataset,
        transforms=transforms_nonchunk,
    )

    opts_dict = vars(opts)
    with open(os.path.join(opts.path_run_dir, 'train_options.json'), 'w') as fo:
        json.dump(opts_dict, fo, indent=4, sort_keys=True)

    kwargs = dict(
        model = model,
        data_provider = data_provider,
        data_provider_nonchunk = data_provider_nonchunk,
        logger = logger,
    )
    kwargs.update(vars(opts))
    train_model(**kwargs)
    
if __name__ == '__main__':
    main()
