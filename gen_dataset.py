import argparse
import util.data
import util.data.transforms
import os
import subprocess
import util
import numpy as np
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--chan', help='target channel for TTF or source channel for SNM')
parser.add_argument('--path_save', help='path to where dataset should be saved')
parser.add_argument('--path_source', help='path to data directory')
parser.add_argument('--project', choices=['ttf', 'snm'], help='path to data directory')
parser.add_argument('--resize_only', action='store_true', help='transform images only with resizing')
opts = parser.parse_args()

def gen_dataset_2():
    print('***** Generating DataSet *****')
    np.random.seed(666)
    if opts.path_save is None:
        chan = opts.chan.lower().replace(' ', '_')
        path_datadir = os.path.basename(os.path.dirname(opts.path_source))
        path_save = os.path.join('data', 'dataset_saves', '{}_{}_{}.ds'.format(opts.project, chan, path_datadir))
    else:
        path_save = opts.path_save
    train_select = True
    if opts.project == 'ttf':
        file_tags = ('_trans.tif', '_{}.tif'.format(opts.chan))
    elif opts.project == 'snm':
        file_tags = ('_{}.tif'.format(opts.chan), '_nuc.tif', '_cell.tif')
    else:
        raise NotImplementedError
    train_set_size = 15
    percent_test = None
    # aiming for 0.3 um/px
    z_fac = 0.97
    xy_fac = 0.36
    resize_factors = (z_fac, xy_fac, xy_fac)
    resizer = util.data.transforms.Resizer(resize_factors)
    # capper = util.data.transforms.Capper(std_hi=9)
    if opts.resize_only:
        signal_transforms = resizer
        target_transforms = resizer
    else:
        signal_transforms = (resizer, util.data.transforms.sub_mean_norm)
        target_transforms = (resizer, util.data.transforms.sub_mean_norm)
        # target_transforms = (resizer, util.data.transforms.sub_mean_norm, capper)
    transforms = (signal_transforms, target_transforms)
    # transforms = None  # DEBUG
    dataset = util.data.DataSet2(
        path_save=path_save,
        path_csv=opts.path_source,
        train_select=True,
        task=opts.project,
        chan=opts.chan,
        train_set_size=15, percent_test=None,
        transforms=transforms
    )
    print(dataset)

    # try retrieving first element from dataset
    data = dataset[0]
    if opts.project == 'ttf':
        assert len(data) == 2
    df = dataset.get_active_set()
    max_ele = min(20, len(df))
    for i in range(max_ele):
        path_signal_base = os.path.basename(df.iloc[i, 0])
        path_target_base = os.path.basename(df.iloc[i, 1])
        print('{:4d} | {:50s} | {:50s}'.format(df.index[i], path_signal_base, path_target_base))
    
def gen_dataset():
    print('***** Generating DataSet *****')
    np.random.seed(666)
    if opts.path_save is None:
        path_source_basename = os.path.basename(opts.path_source)
        path_save = os.path.join('data', 'dataset_saves', '{}_{}_{}.ds'.format(opts.project, opts.chan, path_source_basename))
    else:
        path_save = opts.path_save
    train_select = True
    if opts.project == 'ttf':
        file_tags = ('_trans.tif', '_{}.tif'.format(opts.chan))
    elif opts.project == 'snm':
        file_tags = ('_{}.tif'.format(opts.chan), '_nuc.tif', '_cell.tif')
    else:
        raise NotImplementedError
    train_set_size = 15
    percent_test = None
    # aiming for 0.3 um/px
    z_fac = 0.97
    xy_fac = 0.36
    resize_factors = (z_fac, xy_fac, xy_fac)
    resizer = util.data.transforms.Resizer(resize_factors)
    capper = util.data.transforms.Capper(std_hi=9)
    if RESIZE_ONLY:
        signal_transforms = resizer
        target_transforms = resizer
    else:
        signal_transforms = (resizer, util.data.transforms.sub_mean_norm)
        # target_transforms = (resizer, util.data.transforms.sub_mean_norm)
        target_transforms = (resizer, util.data.transforms.sub_mean_norm, capper)
    transforms = (signal_transforms, target_transforms)
    dataset = util.data.DataSet(path_save=path_save,
                                path_source=opts.path_source,
                                train_select=train_select,
                                file_tags=file_tags,
                                train_set_size=train_set_size, percent_test=percent_test,
                                transforms=transforms)
    print(dataset)
    # try retrieving first element from dataset
    data = dataset[0]
    assert len(data) == len(file_tags)
    for f in dataset.get_active_set()[:10]:
        print(f)
    # util.save_img_np(data[0], 'test_output/signal.tif')
    # util.save_img_np(data[1], 'test_output/target.tif')
    
def combine_dataset():
    print('***** testing DataSet *****')
    path_combo = 'data/no_hots_combo'
    if False:
        path_a = 'data/no_hots'
        path_b = 'data/no_hots_2'

        folder_list_a = [i.path for i in os.scandir(path_a) if i.is_dir()]  # order is arbitrary
        folder_list_b = [i.path for i in os.scandir(path_b) if i.is_dir()]  # order is arbitrary
        folder_list_combo = folder_list_a + folder_list_b
        print(len(folder_list_combo), 'folders!')
        for folder in folder_list_combo:
            path_rel_target = os.path.join('..', *folder.split('/')[-2:])
            link_name = os.path.join(path_combo, os.path.basename(folder))
            cmd = 'ln -s {} {}'.format(path_rel_target, link_name)
            print(cmd)
            subprocess.check_call(cmd, shell=True)
        return
    
    train_select = True
    train_set_limit = 30
    # aiming for 0.3 um/px
    z_fac = 0.97
    xy_fac = 0.36
    resize_factors = (z_fac, xy_fac, xy_fac)
    dataset = util.data.DataSet(path_combo, train=train_select, force_rebuild=True, train_set_limit=train_set_limit,
                                transform=util.data.transforms.Resizer(resize_factors))
    print(dataset)

if __name__ == '__main__':
    # gen_dataset()
    gen_dataset_2()
