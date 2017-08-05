import argparse
import os
from aicsimage.io import omeTifReader
import numpy as np
import pandas as pd
import shutil
import subprocess
import util.data
import util.data.transforms
import matplotlib.pyplot as plt
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--chan', choices=['trans', 'dna', 'memb', 'struct'], default='trans', help='target channel for analysis')
parser.add_argument('--n_max', type=int, help='maximum number of images to analyze')
parser.add_argument('--path_source', help='path to data directory or list of file names')
opts = parser.parse_args()

def archive():
    path_base = 'data/tubulin_nobgsub'
    folder_path_list = [i.path for i in os.scandir(path_base) if i.is_dir()]  # order is arbitrary
    folder_path_list.sort()
    idx_start = 49
    idx_end = idx_start + 47
    folder_list_sub = folder_path_list[idx_start:idx_end]
    n_images = len(folder_list_sub)
    print(n_images, 'images')
    for i, folder in enumerate(folder_list_sub):
        print('{:02d}'.format(i), folder)
        
    min_ar = np.zeros(n_images)
    max_ar = np.zeros(n_images)
    median_ar = np.zeros(n_images)
    for i, folder_path in enumerate(folder_list_sub):
        path_dna_tiff = [i.path for i in os.scandir(folder_path) if i.name.endswith('dna.tif')][0]
        fin = omeTifReader.OmeTifReader(path_dna_tiff)
        img_dna = fin.load().astype(np.float32)[0, ]
        min_img = np.amin(img_dna)
        max_img = np.amax(img_dna)
        median_img = np.median(img_dna)
        print(min_img, max_img, median_img)
        min_ar[i] = min_img
        max_ar[i] = max_img
        median_ar[i] = median_img
    print(min_ar)
    print(max_ar)
    print(median_ar)

    for thresh in range(500, 3000, 100):
        count = np.count_nonzero(max_ar < thresh)
        print('thresh: {:4d} | count: {:d}'.format(thresh, count))

    # move items
    thresh = 900
    dir_target = 'data/no_hots_2'
    for i, path_src in enumerate(folder_list_sub):
        if max_ar[i] < thresh:
            base = os.path.basename(path_src)
            path_target = os.path.join(dir_target, base)
            
            cmd_str = 'cp -r {} {}'.format(path_src, path_target)
            print(cmd_str)
            subprocess.check_call(cmd_str, shell=True)

def set_plot_style():
    plt.rc('figure', figsize=(18, 8))
    plt.rc('axes', labelsize='x-large', titlesize='x-large')
    plt.rc('xtick', labelsize='large')
    plt.rc('ytick', labelsize='large')

def plot_mean_intensity_per_slice(dataset, slice_dim='z'):
    idx_dim = 'zyx'.find(slice_dim)
    axes_other = tuple(i for i in range(3) if (i != idx_dim))
    print(axes_other)
    assert idx_dim >= 0
    for i, element in enumerate(dataset):
        plot_kwargs = dict(
            title=dataset.get_name(i),
            xlabel=slice_dim,
            ylabel='mean pixel intensity'
        )
        data = element[0]
        n_means = data.shape[idx_dim]
        slice_vals = range(n_means)
        means = np.mean(data, axis=axes_other)
        fig, ax = plt.subplots(1)
        fig.set_size_inches((12, 8))
        ax.scatter(slice_vals, means)
        ax.set(**plot_kwargs)
        if opts.n_max is not None and (i + 1) == opts.n_max:
            break
    plt.show()

def gen_bxp_dict(ar):
    """Generate dict for matplotlib bxp function."""
    bxp_dict = {}
    bxp_dict['med'] = np.median(ar)
    bxp_dict['q1'] = np.percentile(ar, 25.0)
    bxp_dict['q3'] = np.percentile(ar, 75.0)
    bxp_dict['whislo'] = np.amin(ar)
    bxp_dict['whishi'] = np.amax(ar)

    # optional
    bxp_dict['mean'] = np.mean(ar)
    return bxp_dict

def count_outliers(ar, bxp_dict):
    fac = 5
    iqr = bxp_dict['q3'] - bxp_dict['q1']
    count = np.count_nonzero(np.logical_or(ar < bxp_dict['q1'] - fac*iqr, ar > bxp_dict['q3'] + fac*iqr))
    return count
    
def plot_boxplots(dataset):
    if opts.n_max is not None:
        indices = np.round(np.linspace(0, len(dataset) - 1, opts.n_max)).astype(int)
    else:
        indices = range(len(dataset))
    boxes = []
    legend_str = []
    legend_str.append('***** Legend *****')
    for i, idx in enumerate(indices):
        ar = dataset[idx][0]
        bxp_dict = gen_bxp_dict(ar)
        boxes.append(bxp_dict)
        legend_str.append('{:02d}: {:s} {:d}'.format(i + 1, dataset.get_name(idx), count_outliers(ar, bxp_dict)))
    fig, ax = plt.subplots(1)
    plot_kwargs = dict(
        title='per-image pixel intensity',
        ylabel='mean pixel intensity'
    )
    ax.set(**plot_kwargs)
    ax.bxp(boxes, showfliers=False, showmeans=True)
    plt.show()
    legend = os.linesep.join(legend_str)
    print(legend)
    # flat_images = []
    # for i, idx in enumerate(indices):
    #     data = dataset[idx][0]
    #     flat_image = data.flatten()
    #     flat_images.append(data.flatten())
    #     ax.boxplot([flat_image], positions=[i], showmeans=True, showfliers=False)

def legacy_check_dataset(path):
    print('***** Checking dataset from:', path, '*****')
    train_select = True
    dataset = util.data.DataSet(path, train=train_select)
    print(dataset)
    return
    dims_chunk = (32, 208, 208)
    dims_pin = (0, 0, 0)
    dp = util.data.TestImgDataProvider(dataset, dims_chunk=dims_chunk, dims_pin=dims_pin)
    means_signal = np.zeros(len(dp))
    stds_signal = np.zeros(len(dp))
    means_target = np.zeros(len(dp))
    stds_target = np.zeros(len(dp))
    for i, batch in enumerate(dp):
        trans, dna = batch
        means_signal[i] = np.mean(trans)
        stds_signal[i] = np.std(trans)
        means_target[i] = np.mean(dna)
        stds_target[i] = np.std(dna)
        print(means_signal[i], stds_signal[i], '|', means_target[i], stds_target[i])
    print(means_signal)
    print(stds_signal)
    print('mean of means_signal:', np.mean(means_signal))
    print('mean of stds_signal:', np.mean(stds_signal))
    print('mean of means_target:', np.mean(means_target))
    print('mean of stds_target:', np.mean(stds_target))

def read_csv():
    assert opts.path_source.endswith('.csv')
    assert os.path.isfile(opts.path_source)
    print('path_source:', opts.path_source)
    df = pd.read_csv(opts.path_source)
    pdb.set_trace()
    
            
def main():
    path_source_basename = os.path.basename(opts.path_source)
    path_save = os.path.join('data', 'dataset_saves', 'TMP_{}_{}.ds'.format(opts.chan, path_source_basename))
    file_tags = ('_{}.tif'.format(opts.chan), )
    
    train_select = True
    percent_test = 0
    transforms = None
    transforms = ((util.data.transforms.sub_mean_norm, util.data.transforms.Capper(std_hi=10)),
    )
    dataset = util.data.DataSet(path_save=path_save,
                                path_source=opts.path_source,
                                train_select=train_select,
                                file_tags=file_tags,
                                percent_test=percent_test,
                                transforms=transforms)
    print(dataset)
    set_plot_style()
    plot_boxplots(dataset)
    # plot_mean_intensity_per_slice(dataset)


if __name__ == '__main__':
    # main()
    read_csv()
    # test()
    
