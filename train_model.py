from fnet.utils.general_utils import str_to_class
import argparse
import copy
import fnet
import json
import logging
import numpy as np
import os
import pdb  # noqa: F401
import sys
import time
import torch


def get_dataloader(args, n_iter_remaining, validation=False):
    dataset_kwargs = copy.deepcopy(args.dataset_kwargs)
    path_csv = (args.path_dataset_csv if not validation
                else args.path_dataset_val_csv)
    if path_csv is not None:
        assert 'path_csv' not in dataset_kwargs, 'dataset csv specified twice'
        dataset_kwargs['path_csv'] = path_csv
    ds = str_to_class(args.dataset_class)(**dataset_kwargs)
    bpds_kwargs = copy.deepcopy(args.bpds_kwargs)
    assert 'dataset' not in bpds_kwargs
    if not validation:
        bpds_kwargs['npatches'] = n_iter_remaining*args.batch_size
    else:
        bpds_kwargs['buffer_size'] = len(ds)
        bpds_kwargs['buffer_switch_frequency'] = -1
        bpds_kwargs['npatches'] = 4*args.batch_size
    print(bpds_kwargs)
    bpds = fnet.data.BufferedPatchDataset(dataset=ds, **bpds_kwargs)
    dataloader = torch.utils.data.DataLoader(
        bpds,
        batch_size=args.batch_size,
    )
    return dataloader


def get_loss_val(model, dataloader_val):
    if dataloader_val is None:
        return None
    criterion_val = torch.nn.MSELoss()
    loss_val_sum = 0
    for idx_val, (signal_val, target_val) in enumerate(dataloader_val):
        pred_val = model.predict(signal_val)
        loss_val_batch = criterion_val(pred_val, target_val).item()
        loss_val_sum += loss_val_batch
        print('  loss_val_batch: {:.3f}'.format(loss_val_batch))
    return loss_val_sum/len(dataloader_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
    parser.add_argument('--bpds_kwargs', type=json.loads, default={}, help='kwargs to be passed to BufferedPatchDataset')
    parser.add_argument('--dataset_class', default='fnet.data.CziDataset', help='Dataset class')
    parser.add_argument('--dataset_kwargs', type=json.loads, default={}, help='kwargs to be passed to Dataset class')
    parser.add_argument('--fnet_model_class', default='fnet.models.Model', help='FnetModel class')
    parser.add_argument('--fnet_model_kwargs', type=json.loads, default={}, help='kwargs to be passed to fnet model class')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
    parser.add_argument('--interval_checkpoint', type=int, default=50000, help='intervals at which to save checkpoints of model')
    parser.add_argument('--interval_save', type=int, default=500, help='iterations between saving log/model')
    parser.add_argument('--iter_checkpoint', nargs='+', type=int, default=[], help='iterations at which to save checkpoints of model')
    parser.add_argument('--n_iter', type=int, default=50000, help='number of training iterations')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_dataset_val_csv', type=str, help='path to csv for constructing validation Dataset (evaluated everytime the model is saved)')
    parser.add_argument('--path_run_dir', default='saved_models', help='base directory for saved models')
    parser.add_argument('--seed', type=int, help='random seed')
    args = parser.parse_args()

    time_start = time.time()
    if not os.path.exists(args.path_run_dir):
        os.makedirs(args.path_run_dir)
    if len(args.iter_checkpoint) > 0 or args.interval_checkpoint is not None:
        path_checkpoint_dir = os.path.join(args.path_run_dir, 'checkpoints')
        if not os.path.exists(path_checkpoint_dir):
            os.makedirs(path_checkpoint_dir)

    path_options = os.path.join(args.path_run_dir, 'train_options.json')
    with open(path_options, 'w') as fo:
        json.dump(vars(args), fo, indent=4, sort_keys=True)

    # Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        os.path.join(args.path_run_dir, 'run.log'), mode='a'
    )
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Instantiate Model
    path_model = os.path.join(args.path_run_dir, 'model.p')
    model = fnet.models.load_or_init_model(path_model, path_options)
    model.to_gpu(args.gpu_ids)
    logger.info(model)

    path_losses_csv = os.path.join(args.path_run_dir, 'losses.csv')
    if os.path.exists(path_losses_csv):
        fnetlogger = fnet.FnetLogger(path_losses_csv)
        logger.info('History loaded from: {:s}'.format(path_losses_csv))
    else:
        fnetlogger = fnet.FnetLogger(
            columns=['num_iter', 'loss_batch', 'loss_val']
        )

    n_remaining_iterations = max(0, (args.n_iter - model.count_iter))
    dataloader_train = get_dataloader(args, n_remaining_iterations)
    dataloader_val = get_dataloader(
        args, n_remaining_iterations, validation=True
    )
    for i, (signal, target) in enumerate(dataloader_train, model.count_iter):
        do_save = ((i + 1) % args.interval_save == 0) or \
                  ((i + 1) == args.n_iter)
        loss_batch = model.do_train_iter(signal, target)
        loss_val = get_loss_val(model, dataloader_val) if do_save else None
        fnetlogger.add(
            {'num_iter': i + 1, 'loss_batch': loss_batch, 'loss_val': loss_val}
        )
        print('num_iter: {:6d} | loss_batch: {:.3f} | loss_val: {}'.format(
            i + 1, loss_batch, loss_val
        ))
        if do_save:
            model.save(path_model)
            fnetlogger.to_csv(path_losses_csv)
            logger.info('BufferedPatchDataset buffer history: {}'.format(dataloader_train.dataset.get_buffer_history()))
            logger.info('loss log saved to: {:s}'.format(path_losses_csv))
            logger.info('model saved to: {:s}'.format(path_model))
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
        if ((i + 1) in args.iter_checkpoint) or \
           ((i + 1) % args.interval_checkpoint == 0):
            path_save_checkpoint = os.path.join(path_checkpoint_dir, 'model_{:06d}.p'.format(i + 1))
            model.save(path_save_checkpoint)
            logger.info('model checkpoint saved to: {:s}'.format(path_save_checkpoint))


if __name__ == '__main__':
    main()
