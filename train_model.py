import argparse
import fnet.data
import fnet.data.datasets
import fnet.fnet_model
import json
import logging
import numpy as np
import os
import pandas as pd
import pdb
import shutil
import sys
import time
import torch
import warnings

def main():
    time_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--args_dataset', type=json.loads, default='{}', help='size of each batch')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
    parser.add_argument('--buffer_size', type=int, default=5, help='number of images to cache in memory')
    parser.add_argument('--buffer_switch_frequency', type=int, default=720, help='BufferedPatchDataset buffer switch frequency')
    parser.add_argument('--checkpoint_testing', action='store_true', help='set to test model at checkpoints')
    parser.add_argument('--dataset', type=str, default='dna', help='name of dataset')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
    parser.add_argument('--iter_checkpoint', type=int, default=500, help='iterations between saving log/model checkpoints')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_iter', type=int, default=500, help='number of training iterations')
    parser.add_argument('--nn_module', default='ttf_v8_nn', help='name of neural network module')
    parser.add_argument('--path_run_dir', default='saved_models', help='base directory for saved models')
    parser.add_argument('--replace_interval', type=int, default=-1, help='iterations between replacements of images in cache')
    parser.add_argument('--seed', type=int, help='random seed')
    opts = parser.parse_args()
    
    if not os.path.exists(opts.path_run_dir):
        os.makedirs(opts.path_run_dir)

    #Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.path_run_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    #Set GPU
    main_gpu_id = opts.gpu_ids if isinstance(opts.gpu_ids, int) else opts.gpu_ids[0]
    torch.cuda.set_device(main_gpu_id)
    logger.info('main GPU ID: {:d}'.format(torch.cuda.current_device()))

    #Set random seed
    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    #Instantiate Model
    model = fnet.fnet_model.Model(
        nn_module=opts.nn_module,
        lr=opts.lr,
        gpu_ids=opts.gpu_ids,
    )
    logger.info('Model instianted from: {:s}'.format(opts.nn_module))
    
    #Load saved model if it already exists
    path_model = os.path.join(opts.path_run_dir, 'model.p')
    if os.path.exists(path_model):
        model.load_state(path_model)
        logger.info('model loaded from: {:s}'.format(path_model))
    logger.info(model)

    #Load saved history if it already exists
    path_losses_csv = os.path.join(opts.path_run_dir, 'losses.csv')
    df_losses = pd.DataFrame()
    if os.path.exists(path_model):
        df_losses = pd.read_csv(path_losses_csv)
        logger.info('History loaded from: {:s}'.format(path_losses_csv))

    ds = getattr(fnet.data.datasets, 'load_' + opts.dataset)()
    ds_patch = fnet.data.BufferedPatchDataset(
        dataset = ds,
        patch_size = [32, 64, 64],
        buffer_size = opts.buffer_size,
        buffer_switch_frequency = opts.buffer_switch_frequency,
        npatches = max(0, (opts.n_iter - model.count_iter)*opts.batch_size),
        verbose = True,
    )
    dataloader_train = torch.utils.data.DataLoader(
        ds_patch,
        batch_size = opts.batch_size,
    )

    dataloader_test = None
    if opts.checkpoint_testing:
        raise NotImplementedError
        ds = getattr(fnet.data.datasets, 'load_' + opts.dataset)(
            train = False,
        )
        dataloader_test = torch.utils.data.DataLoader(
            ds,
            batch_size = 1,
        )
    with open(os.path.join(opts.path_run_dir, 'train_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)

    for i, (signal, target) in enumerate(dataloader_train, model.count_iter):
        loss_batch = model.do_train_iter(signal, target)
        logger.info('num_iter: {:4d} | loss_batch: {:.4f}'.format(i + 1, loss_batch))
        dict_iter = dict(
            num_iter = i + 1,
            loss_batch = loss_batch,
        )
        df_losses_curr = pd.concat([df_losses, pd.DataFrame([dict_iter])], ignore_index=True)
        if ((i + 1) % opts.iter_checkpoint == 0) or ((i + 1) == opts.n_iter):
            if dataloader_test is not None:
                raise NotImplementedError
            model.save_state(path_model)
            df_losses_curr.to_csv(path_losses_csv, index=False)
            logger.info('model saved to: {:s}'.format(path_model))
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
        df_losses = df_losses_curr
    logger.info('total training time: {:.1f} s'.format(time.time() - time_start))

if __name__ == '__main__':
    main()
