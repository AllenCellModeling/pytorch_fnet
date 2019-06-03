"""Trains a model."""


from pathlib import Path
from typing import Callable, Dict, Optional
import argparse
import copy
import datetime
import json
import logging
import os
import pprint
import time

import numpy as np
import torch

from fnet.cli.init import save_default_train_options
from fnet.data import BufferedPatchDataset
from fnet.utils.general_utils import add_logging_file_handler
from fnet.utils.general_utils import str_to_object
import fnet
import fnet.utils.viz_utils as vu


logger = logging.getLogger(__name__)


def log_training_options(options: Dict) -> None:
    """Logs training options."""
    for line in (
            ['*** Training options ***']
            + pprint.pformat(options).split(os.linesep)
    ):
        logger.info(line)


def set_seeds(seed: Optional[int]) -> None:
    """Sets random seeds"""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_cuda(gpu: int) -> None:
    """Initialize Pytorch CUDA state."""
    if gpu < 0:
        return
    try:
        torch.cuda.set_device(gpu)
        torch.cuda.init()
    except RuntimeError:
        logger.exception('Failed to init CUDA')


def get_bpds_train(args: argparse.Namespace) -> BufferedPatchDataset:
    """Creates data provider for training."""
    ds_fn = str_to_object(args.dataset_train)
    if not isinstance(ds_fn, Callable):
        raise ValueError('Dataset function should be Callable')
    ds = ds_fn(**args.dataset_train_kwargs)
    return BufferedPatchDataset(dataset=ds, **args.bpds_kwargs)


def get_bpds_val(args: argparse.Namespace) -> Optional[BufferedPatchDataset]:
    """Creates data provider for validation."""
    if args.dataset_val is None:
        return None
    bpds_kwargs = copy.deepcopy(args.bpds_kwargs)
    ds_fn = str_to_object(args.dataset_val)
    if not isinstance(ds_fn, Callable):
        raise ValueError('Dataset function should be Callable')
    ds = ds_fn(**args.dataset_val_kwargs)
    bpds_kwargs['buffer_size'] = min(4, len(ds))
    bpds_kwargs['buffer_switch_interval'] = -1
    return BufferedPatchDataset(dataset=ds, **bpds_kwargs)


def add_parser_arguments(parser) -> None:
    """Add training script arguments to parser."""
    parser.add_argument('json', help='json with training options')
    parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help='gpu_id(s)')


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Trains a model."""
    time_start = time.time()
    if args is None:
        parser = argparse.ArgumentParser()
        add_parser_arguments(parser)
        args = parser.parse_args()
    if not os.path.exists(args.json):
        save_default_train_options(args.json)
        return
    with open(args.json, 'r') as fi:
        train_options = json.load(fi)
    args.__dict__.update(train_options)
    add_logging_file_handler(Path(args.path_save_dir, 'train_model.log'))
    logger.info('Started training at: %s', datetime.datetime.now())

    set_seeds(args.seed)
    log_training_options(vars(args))
    path_model = os.path.join(args.path_save_dir, 'model.p')
    model = fnet.models.load_or_init_model(path_model, args.json)
    init_cuda(args.gpu_ids[0])
    model.to_gpu(args.gpu_ids)
    logger.info(model)

    path_losses_csv = os.path.join(args.path_save_dir, 'losses.csv')
    if os.path.exists(path_losses_csv):
        fnetlogger = fnet.FnetLogger(path_losses_csv)
        logger.info('History loaded from: {:s}'.format(path_losses_csv))
    else:
        fnetlogger = fnet.FnetLogger(
            columns=['num_iter', 'loss_train', 'loss_val']
        )

    bpds_train = get_bpds_train(args)
    bpds_val = get_bpds_val(args)

    for idx_iter in range(model.count_iter, args.n_iter):
        x_batch, y_batch = bpds_train.get_batch(args.batch_size)
        do_save = ((idx_iter + 1) % args.interval_save == 0) or \
                  ((idx_iter + 1) == args.n_iter)
        loss_train = model.train_on_batch(x_batch, y_batch)
        loss_val = None
        if do_save and bpds_val is not None:
            loss_val = model.test_on_iterator(
                [bpds_val.get_batch(args.batch_size) for _ in range(4)]
            )
        fnetlogger.add(
            {
                'num_iter': idx_iter + 1,
                'loss_train': loss_train,
                'loss_val': loss_val,
            }
        )
        print(
            f'iter: {fnetlogger.data["num_iter"][-1]:6d} | '
            f'loss_train: {fnetlogger.data["loss_train"][-1]:.4f}'
        )
        if do_save:
            model.save(path_model)
            fnetlogger.to_csv(path_losses_csv)
            logger.info(
                'BufferedPatchDataset buffer history: %s',
                bpds_train.get_buffer_history(),
            )
            logger.info('loss log saved to: {:s}'.format(path_losses_csv))
            logger.info('model saved to: {:s}'.format(path_model))
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
        if ((idx_iter + 1) in args.iter_checkpoint) or \
           ((idx_iter + 1) % args.interval_checkpoint == 0):
            path_checkpoint = os.path.join(
                args.path_save_dir,
                'checkpoints',
                'model_{:06d}.p'.format(idx_iter + 1),
            )
            model.save(path_checkpoint)
            logger.info('Saved model checkpoint: %s', path_checkpoint)
            vu.plot_loss(
                args.path_save_dir,
                path_save=os.path.join(args.path_save_dir, 'loss_curves.png'),
            )


if __name__ == '__main__':
    main()
