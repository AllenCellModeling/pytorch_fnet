import argparse
import os
from aicsimage.io import omeTifReader
import numpy as np
import pandas as pd
import shutil
import subprocess
import util
import util.data
import util.data.transforms
import matplotlib.pyplot as plt
import pdb
from util.data.czireader import CziReader

parser = argparse.ArgumentParser()
parser.add_argument('--element_num', default=0, choices=[0, 1, 2], type=int, help='index of dataset element to analyze')
parser.add_argument('--n_max', type=int, help='maximum number of images to analyze')
parser.add_argument('--path_dataset', help='path to data directory or list of file names')
parser.add_argument('--path_source', help='path to data file/directory')
parser.add_argument('--plot_sel', choices=['boxes', 'means'], default='means', help='path to data file/directory')
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
    plt.rc('figure.subplot', wspace=0.05)
    plt.rc('axes', labelsize='x-large', titlesize='x-large')
    plt.rc('xtick', labelsize='large')
    plt.rc('ytick', labelsize='large')

def plot_mean_intensity_per_slice(dataset, slice_dim='z'):
    def means_ok(means):
        """"""
        median_of_means = np.median(means)
        # print('DEBUG: median_of_means', median_of_means)
        # print(means)
        if np.any(means < median_of_means*0.5):
            return False
        return True
    
    idx_dim = 'zyx'.find(slice_dim)
    axes_other = tuple(i for i in range(3) if (i != idx_dim))
    print('DEBUG: other axes', axes_other)
    assert idx_dim >= 0
    
    if opts.n_max is not None:
        indices = np.round(np.linspace(0, len(dataset) - 1, opts.n_max)).astype(int)
    else:
        indices = range(len(dataset))
    
    report_str = []
    report_str.append('***** REPORT *****')
    for i, idx in enumerate(indices):
        ar = dataset[idx][opts.element_num]
        print('DEBUG: shape', ar.shape)
        if ar.shape[0] < 32:
            covfefe
        n_means = ar.shape[idx_dim]
        slice_vals = range(n_means)
        means = np.mean(ar, axis=axes_other)
        fig, ax = plt.subplots(1)
        # fig.set_size_inches((12, 8))
        ax.scatter(slice_vals, means)
        y_low = ax.get_ylim()[1]*0.1
        if y_low > 0:
            y_low *= -1
        plot_kwargs = dict(
            title=dataset.get_name(idx, opts.element_num),
            xlabel=slice_dim,
            ylabel='mean pixel intensity',
            ylim=(y_low, ax.get_ylim()[1]*1.1)
        )
        ax.set(**plot_kwargs)
        status = '  pass  ' if means_ok(means) else '* FAIL *'
        report_str.append('{} | {}'.format(status, dataset.get_name(idx, opts.element_num)))
    plt.show()
    print(os.linesep.join(report_str))
    

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
        ar = dataset[idx][opts.element_num]
        bxp_dict = gen_bxp_dict(ar)
        boxes.append(bxp_dict)
        legend_str.append('{:02d}: {:s} {:d}'.format(i + 1, dataset.get_name(idx, opts.element_num), count_outliers(ar, bxp_dict)))
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

def test_read_csv():
    path_test = 'data/dataset_saves/TMP_test.ds'
    np.random.seed(666)
    struct = 'Lamin B1'
    dataset = util.data.DataSet2(
        path_save=path_test,
        path_csv=opts.path_source,
        train_select=True,
        task='ttf',
        chan=struct,
        train_set_size=15, percent_test=None,
        transforms=None
    )
    df_csv = dataset.get_df_csv()
    mask = df_csv['structureProteinName'] == struct
    mask = mask & (df_csv['inputFolder'].str.contains('aics/microscopy'))
    df = df_csv[mask]

    col_chans = []
    for col in df_csv.columns:
        if 'Channel' in col:
            col_chans.append(col)
    pdb.set_trace()

def display_czi_channels(z_display=None):
    path_pre = opts.path_source
    if 'allen/aics' in path_pre:
        path = os.path.join('/root/data', path_pre.split('allen/aics/')[-1])
    else:
        path = path_pre
    assert os.path.isfile(path)
    czi = CziReader(path)
    if z_display is None:
        size_z = czi.get_size('Z')
        z_iter = tuple(int(i*size_z) for i in [0.25, 0.5, 0.75])
    elif isinstance(z_display, int):
        z_iter = (z_display, )
    else:
        z_iter = tuple(z_display)
    for chan in range(czi.get_size('C')):
    # for chan in [0, 1]:
        fig, ax = plt.subplots(ncols=len(z_iter))
        vol = czi.get_volume(chan)
        for idx_ax, z in enumerate(z_iter):
            ax[idx_ax].imshow(vol[z, :, :], cmap='gray', interpolation='bilinear')
            plot_kwargs = dict(
                title='chan: {} | z: {}'.format(chan, z)
            )
            ax[idx_ax].set(**plot_kwargs)
            ax[idx_ax].axis('off')
    plt.show()

def main():
    dataset = util.data.DataSet2(
        path_load=opts.path_dataset,
        train_select=True
    )
    print(dataset)
    if opts.plot_sel == 'boxes':
        plot_boxplots(dataset)
    if opts.plot_sel == 'means':
        plot_mean_intensity_per_slice(dataset, slice_dim='z')

if __name__ == '__main__':
    set_plot_style()
    main()
    # display_czi_channels()
    # test_read_csv()
    # test()
    
