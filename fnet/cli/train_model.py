from fnet.cli.init import save_default_train_options
from typing import Callable, Optional
from fnet.utils.general_utils import str_to_object
import argparse
import inspect
import copy
import fnet
import fnet.utils.viz_utils as vu
import json
import logging
import numpy as np
import os
import pdb
import pprint
import sys
import time
import torch
import torch.utils.data


def init_cuda(gpu: int) -> None:
    """Initialize Pytorch CUDA state."""
    if gpu < 0:
        return
    torch.cuda.set_device(gpu)
    torch.cuda.init()


def get_dataloaders(
        args: argparse.Namespace,
        n_iter_remaining: int,
        validation: bool = False,
) -> Optional[torch.utils.data.DataLoader]:
    """Creates DataLoader objects from specified dataset."""
    assert 'dataset' not in args.bpds_kwargs
    bpds_kwargs = copy.deepcopy(args.bpds_kwargs)
    bpds_kwargs['npatches'] = n_iter_remaining*args.batch_size
    ds_fn = str_to_object(args.dataset)
    assert isinstance(ds_fn, Callable)
    if 'train' not in inspect.getfullargspec(ds_fn).args and validation:
        # If 'train' if not a dataset function parameter, then assume there is
        # no validation set.
        return None
    ds = ds_fn(train=not validation, **args.dataset_kwargs)
    if validation:
        bpds_kwargs['buffer_size'] = 4
        bpds_kwargs['buffer_switch_frequency'] = -1
        bpds_kwargs['npatches'] = 16*args.batch_size
    print('bpds_kwargs', bpds_kwargs)
    bpds = fnet.data.BufferedPatchDataset(
        dataset=ds, **bpds_kwargs
    )
    dataloader = torch.utils.data.DataLoader(
        bpds,
        batch_size=args.batch_size,
    )
    return dataloader


def init_logger(path_save: str) -> None:
    """Initialize training logger.

    Parameters
    ----------
    path_save
        Location to save training log.

    """
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path_save, mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def add_parser_arguments(parser) -> None:
    """Add training script arguments to parser."""
    parser.add_argument('json', help='json with training options')
    parser.add_argument(
        '--gpu_ids', nargs='+', default=[-1], type=int, help='gpu_id(s)'
    )


def main(args: Optional[argparse.Namespace]) -> None:
    """Trains a model."""
    time_start = time.time()
    if args is None:
        parser = argparse.ArgumentParser()
        add_parser_arguments(parser)
        args = parse_args()
    if not os.path.exists(args.json):
        save_default_train_options(args.json)
        return
    with open(args.json, 'r') as fi:
        train_options = json.load(fi)
    args.__dict__.update(train_options)
    print('*** Training options ***')
    pprint.pprint(args.__dict__)

    # Make checkpoint directory if necessary
    if len(args.iter_checkpoint) > 0 or args.interval_checkpoint is not None:
        path_checkpoint_dir = os.path.join(args.path_save_dir, 'checkpoints')
        if not os.path.exists(path_checkpoint_dir):
            os.makedirs(path_checkpoint_dir)
    logger = init_logger(path_save=os.path.join(args.path_save_dir, 'run.log'))

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Instantiate Model
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

    n_remaining_iterations = max(0, (args.n_iter - model.count_iter))
    dataloader_train = get_dataloaders(args, n_remaining_iterations)
    dataloader_val = get_dataloaders(
        args, n_remaining_iterations, validation=True,
    )
    for idx_iter, (x_batch, y_batch) in enumerate(
            dataloader_train, model.count_iter
    ):
        do_save = ((idx_iter + 1) % args.interval_save == 0) or \
                  ((idx_iter + 1) == args.n_iter)
        loss_train = model.train_on_batch(x_batch, y_batch)
        loss_val = None
        if do_save and dataloader_val is not None:
            loss_val = model.test_on_iterator(dataloader_val)
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
            logger.info('BufferedPatchDataset buffer history: {}'.format(dataloader_train.dataset.get_buffer_history()))
            logger.info('loss log saved to: {:s}'.format(path_losses_csv))
            logger.info('model saved to: {:s}'.format(path_model))
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
        if ((idx_iter + 1) in args.iter_checkpoint) or \
           ((idx_iter + 1) % args.interval_checkpoint == 0):
            path_save_checkpoint = os.path.join(path_checkpoint_dir, 'model_{:06d}.p'.format(idx_iter + 1))
            model.save(path_save_checkpoint)
            logger.info(f'Saved model checkpoint: {path_save_checkpoint}')
            vu.plot_loss(
                args.path_save_dir,
                path_save=os.path.join(args.path_save_dir, 'loss_curves.png'),
            )


if __name__ == '__main__':
    main()
