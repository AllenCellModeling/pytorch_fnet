from typing import Optional
import argparse
import json
import os
import shutil
import sys


def save_example_scripts(path_save_dir: str) -> None:
    """Save example training and prediction scripts.

    Parameters
    ----------
    path_save_dir
        Directory in which to save scripts.

    """
    if not os.path.exists(path_save_dir):
        os.makedirs(path_save_dir)
    path_examples_dir = os.path.join(
        os.path.dirname(sys.modules['fnet'].__file__), 'cli'
    )
    for fname in ['train_model.py', 'predict.py']:
        path_src = os.path.join(path_examples_dir, fname)
        path_dst = os.path.join(path_save_dir, fname)
        if os.path.exists(path_dst):
            print('Example script already exists:', path_dst)
            continue
        shutil.copy(path_src, path_dst)
        print('Saved:', path_dst)


def save_default_train_options(path_save: str) -> None:
    """Save default training options json.

    Parameters
    ----------
    path_save
        Save path for default training options json.

    """
    if os.path.exists(path_save):
        print('Training options file already exists:', path_save)
        return
    dirname = os.path.dirname(path_save)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    train_options = {
        'batch_size': 28,
        'bpds_kwargs': {
            'buffer_size': 16,
            'buffer_switch_frequency': 2800,  # every 100 updates
            'patch_size': [32, 64, 64]
        },
        'dataset': 'aics_x',
        'dataset_kwargs': {},
        'fnet_model_class': 'fnet.fnet_model.Model',
        'fnet_model_kwargs': {
            'betas': [0.9, 0.999],
            'criterion_class': 'torch.nn.MSELoss',
            'init_weights': False,
            'lr': 0.001,
            'nn_class': 'fnet.nn_modules.fnet_nn_3d.Net',
            'scheduler': None,
        },
        'interval_checkpoint': 50000,
        'interval_save': 1000,
        'iter_checkpoint': [],
        'n_iter': 250000,
        'path_save_dir': 'saved_models/test',
        'seed': None,
    }
    with open(path_save, 'w') as fo:
        json.dump(train_options, fo, indent=4, sort_keys=True)
        print('Saved:', path_save)


def add_parser_arguments(parser: argparse.ArgumentParser) -> None:
    """Add init script arguments to parser."""
    parser.add_argument(
        '--path_scripts_dir',
        default='scripts',
        help='Path to where example scripts should be saved.'
    )
    parser.add_argument(
        '--path_train_template',
        default='train_options_templates/default.json',
        help='Path to where training options template should be saved.'
    )


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Install default training options and example model training/prediction
    scripts into current directory."""
    if args is None:
        parser = argparse.ArgumentParser()
        add_parser_arguments(parser)
        args = parse_args()
    save_example_scripts(args.path_scripts_dir)
    save_default_train_options(args.path_train_template)
