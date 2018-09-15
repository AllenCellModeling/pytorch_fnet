import argparse
import fnet.data
import fnet.fnet_model
import json
import logging
import numpy as np
import os
import pdb
import sys
import time
import torch
import warnings
import importlib

def get_dataloader(remaining_iterations, opts, validation=False, ds = None):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    if ds is None:
        ds = getattr(fnet.data, opts.class_dataset)(
            path_csv = opts.path_dataset_csv if not validation else opts.path_dataset_val_csv,
            transform_source = transform_signal,
            transform_target = transform_target,
            **opts.fds_kwargs
        )
    ds_patch = fnet.data.BufferedPatchDataset(
        dataset = ds,
        patch_size = opts.patch_size,
        buffer_size = opts.buffer_size if not validation else len(ds),
        buffer_switch_frequency = opts.buffer_switch_frequency if not validation else -1,
        npatches = remaining_iterations*opts.batch_size if not validation else 20*opts.batch_size,
        verbose = True,
        shuffle_images = opts.shuffle_images,
        **opts.bpds_kwargs,
    )
    if validation:
        ds_patch.augment_data_rate = 0
    dataloader = torch.utils.data.DataLoader(
        ds_patch,
        batch_size = opts.batch_size,
    )
    return dataloader, ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adaptive', action='store_true', help='set to use adaptive approach')
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
    parser.add_argument('--bpds_kwargs', type=json.loads, default={}, help='kwargs to be passed to BufferedPatchDataset')
    parser.add_argument('--buffer_size', type=int, default=5, help='number of images to cache in memory')
    parser.add_argument('--buffer_switch_frequency', type=int, default=720, help='BufferedPatchDataset buffer switch frequency')
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--fds_kwargs', type=json.loads, default={}, help='kwargs to be passed to FnetDataset')
    parser.add_argument('--source_points', nargs='+', type=int, default=None, help='time points to use')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
    parser.add_argument('--interval_save', type=int, default=1000, help='iterations between saving log/model')
    parser.add_argument('--iter_checkpoint', nargs='+', type=int, default=[], help='iterations at which to save checkpoints of model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--adamw_decay', type=float, default=0, help='adamw decay')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--n_iter', type=int, default=500, help='maximum number of training iterations')
    parser.add_argument('--nn_kwargs', type=json.loads, default={}, help='kwargs to be passed to nn ctor')
    parser.add_argument('--nn_module', default='fnet_nn_3d', help='name of neural network module')
    parser.add_argument('--max_in_channels', type=int, default=5, help='max model size to consider')
    parser.add_argument('--max_lag', type=int, default=10, help='max number of lags to consider')
    parser.add_argument('--criterion_fn', default='torch.nn.MSELoss', help='loss function')
    parser.add_argument('--patch_size', nargs='+', type=int, default=[32, 64, 64], help='size of patches to sample from Dataset elements')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_dataset_val_csv', type=str, help='path to csv for constructing validation Dataset (evaluated everytime the model is saved)')
    parser.add_argument('--path_run_dir', default='saved_models', help='base directory for saved models')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--shuffle_images', action='store_true', help='set to shuffle images in BufferedPatchDataset')
    parser.add_argument('--transform_signal', nargs='+', default=['fnet.transforms.normalize'], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize'], help='list of transforms on Dataset target')
    opts = parser.parse_args()
    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    factor_z = 1/2
    if opts.patch_size[1] == 32:
        factor_yx /= 2
    default_resizer_str = 'fnet.transforms.Resizer(({:f}, {:f}, {:f}))'.format(factor_z, factor_yx, factor_yx)
    opts.transform_signal.append(default_resizer_str)
    opts.transform_target.append(default_resizer_str)
        
    time_start = time.time()
    if not os.path.exists(opts.path_run_dir):
        os.makedirs(opts.path_run_dir)
    if len(opts.iter_checkpoint) > 0:
        path_checkpoint_dir = os.path.join(opts.path_run_dir, 'checkpoints')
        if not os.path.exists(path_checkpoint_dir):
            os.makedirs(path_checkpoint_dir)

    #Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.path_run_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    #Set random seed
    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed_all(opts.seed)

    #Instantiate Model
    module, attr = opts.criterion_fn.rsplit('.',1)
    criterion_fn = getattr(importlib.import_module(module),attr)

    fit_model = True
    best_loss_val = float('inf')
    path_prev_model = None
    dataloader_train = None
    dataloader_val = None
    
    if opts.source_points is not None:
        source_points = eval(opts.source_points)
    else:
        source_points=[opts.fds_kwargs['n_offset']]
    
    while fit_model:
        model = fnet.fnet_model.Model(
            nn_module=opts.nn_module,
            lr=opts.lr,
            weight_decay=opts.weight_decay,
            adamw_decay=opts.adamw_decay,
            lambda_reg=opts.lambda_reg,
            gpu_ids=opts.gpu_ids,
            nn_kwargs=opts.nn_kwargs,
            criterion_fn=criterion_fn,
        )
        logger.info('Model instianted from: {:s}'.format(opts.nn_module))
        
        name_model = 'model_{:s}.p'.format('_'.join([str(i) for i in source_points]))
        path_model = os.path.join(opts.path_run_dir, name_model)

        if os.path.exists(path_model):
            model.load_state(path_model, gpu_ids=opts.gpu_ids)
            logger.info('model loaded from: {:s}'.format(path_model))
        elif path_prev_model is not None:
            model.load_weights(path_prev_model, gpu_ids=opts.gpu_ids)
            logger.info('model weights initialized from: {:s}'.format(path_prev_model))

        logger.info(model)

        #Load saved history if it already exists
        path_losses_csv = os.path.join(opts.path_run_dir, 'losses.csv')
        if os.path.exists(path_losses_csv):
            fnetlogger = fnet.FnetLogger(path_losses_csv)
            logger.info('History loaded from: {:s}'.format(path_losses_csv))
        else:
            fnetlogger = fnet.FnetLogger(columns=['num_iter', 'loss_batch'])
            
        path_losses_val_csv = os.path.join(opts.path_run_dir, 'losses_val.csv')
        if os.path.exists(path_losses_val_csv):
            fnetlogger_val = fnet.FnetLogger(path_losses_val_csv)
            logger.info('History loaded from: {:s}'.format(path_losses_val_csv))
        else:
            fnetlogger_val = fnet.FnetLogger(columns=['num_iter', 'loss_val'])
            
        path_best_val_csv = os.path.join(opts.path_run_dir, 'best_loss_val.csv')
        if os.path.exists(path_best_val_csv):
            fnetlogger_best_val = fnet.FnetLogger(path_best_val_csv)
            logger.info('History loaded from: {:s}'.format(path_best_val_csv))
        else:
            fnetlogger_best_val = fnet.FnetLogger(columns=['name_model', 'best_val'])

        n_remaining_iterations = max(0, (opts.n_iter - model.count_iter))
        if n_remaining_iterations <= 0:
            path_prev_model = path_model
            opts.nn_kwargs["in_channels"] += 1
            opts.fds_kwargs["n_source_points"] += 1
            continue
            
        if dataloader_train is None:
            dataloader_train, ds = get_dataloader(n_remaining_iterations, opts)
            #ds.source_points = source_points
        print(ds.source_points)    
        ds.val_mode = True
        sig_val, target_val = ds.get_prediction_batch(len(ds)-1)
        ds.val_mode = False
            
        #if dataloader_val is None:
            #ds.val_mode = True
            #dataloader_val, ds = get_dataloader(n_remaining_iterations, opts, validation = True, ds = ds)
            #ds.val_mode = False

        with open(os.path.join(opts.path_run_dir, 'train_options.json'), 'w') as fo:
            json.dump(vars(opts), fo, indent=4, sort_keys=True)

        best_loss_val_iter = float('inf')
        name_model_list = fnetlogger_best_val.data['name_model']
        if name_model in name_model_list:
            ix = name_model_list.index(name_model)
            best_loss_val_iter = fnetlogger_best_val.data['best_val'][ix]
        best_loss_val = min(fnetlogger_best_val.data['best_val']) if len(name_model_list) > 0 else float('inf')
        loss_tot = 0
        term_count = 0
        for i, (signal, target) in enumerate(dataloader_train, model.count_iter):
            loss_batch = model.do_train_iter(signal, target)
            fnetlogger.add({'num_iter': i + 1, 'loss_batch': loss_batch})
            print('num_iter: {:6d} | loss_batch: {:.3f}'.format(i + 1, loss_batch))
            dict_iter = dict(
                num_iter = i + 1,
                loss_batch = loss_batch,
            )
            loss_tot += loss_batch
            if ((i + 1) % opts.interval_save == 0) or ((i + 1) == opts.n_iter):
                fnetlogger.to_csv(path_losses_csv)
                logger.info('BufferedPatchDataset buffer history: {}'.format(dataloader_train.dataset.get_buffer_history()))
                logger.info('loss log saved to: {:s}'.format(path_losses_csv))
                logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
                print('loss_avg: {:.3f}'.format(loss_tot/opts.interval_save))
                loss_tot = 0
                criterion_val = model.criterion_fn()

                loss_val_sum = 0

                n_batches = sig_val.size(0)
                sig_shape = [1,ds.n_source_points,16,240,240]
                for b in range(n_batches):
                    sig_b = torch.zeros(sig_shape)
                    sig_b[slice(None),slice(None),slice(None),slice(0,112),slice(0,176)] = sig_val[slice(b, b+1)]
                    #sig_b = sig_val[slice(b,b+1)]
                    pred_val = model.predict(sig_b)[slice(None),slice(0,1),slice(None),slice(0,112),slice(0,176)]
                    #loss_val_sum += float(criterion_val(pred_val, torch.autograd.Variable(target_val[slice(b, b+1)])).data[0])
                    loss_val_sum += torch.mean((pred_val - target_val[slice(b, b+1)]).pow(2))
                
                loss_val_iter = loss_val_sum/n_batches
                print('loss_val: {:.3f}'.format(loss_val_iter))
                fnetlogger_val.add({'num_iter': i + 1, 'loss_val': loss_val_iter})
                fnetlogger_val.to_csv(path_losses_val_csv)
                logger.info('loss val log saved to: {:s}'.format(path_losses_val_csv))
                
                #early stopping criterion to prevent overfitting and speed up training
                #if model improves avg validation loss, save model; otherwise continue fitting
                if loss_val_iter <= best_loss_val_iter:
                    best_loss_val_iter = loss_val_iter
                    term_count = 0
                    model.save_state(path_model)
                    logger.info('model saved to: {:s}'.format(path_model))
                else:
                    term_count += 1
                print('best_loss_val: {:.3f}'.format(best_loss_val_iter))
                if term_count >= 5:
                    break
                
            if (i + 1) in opts.iter_checkpoint:
                path_save_checkpoint = os.path.join(path_checkpoint_dir, 'model_{:06d}.p'.format(i + 1))
                model.save_state(path_save_checkpoint)
                logger.info('model checkpoint saved to: {:s}'.format(path_save_checkpoint))
                
        # if performance on validation dataset improves, update prev model and add a new lag
        # otherwise, initialize next model from smaller model and increment furthest lag by 1
        # first get best validation loss from current model
        fnetlogger_best_val.add({'name_model': name_model, 'best_val': best_loss_val_iter})
        fnetlogger_best_val.to_csv(path_best_val_csv)
        loss_val = best_loss_val_iter
        if not opts.adaptive or loss_val <= best_loss_val:
            best_loss_val = loss_val
            path_prev_model = path_model
            ds.source_points = [ds.source_points[0] + 1] + ds.source_points
            opts.nn_kwargs["in_channels"] += 1
            ds.n_source_points += 1
        else:
            ds.source_points[0] += 1
            
        fit_model = ds.n_source_points <= opts.max_in_channels and ds.source_points[0] <= opts.max_lag
            
        source_points = ds.source_points

if __name__ == '__main__':
    main()
