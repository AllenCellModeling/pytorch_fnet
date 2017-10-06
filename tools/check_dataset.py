import sys
sys.path.append('.')
import argparse
import fnet.data
import numpy as np
import os
import pandas as pd
import pdb
import scipy.misc

def set_plot_style():
    plt.rc('figure', figsize=(18, 8))
    plt.rc('figure.subplot', wspace=0.05)
    plt.rc('axes', labelsize='x-large', titlesize='x-large')
    plt.rc('xtick', labelsize='large')
    plt.rc('ytick', labelsize='large')

def plot_mean_intensity_per_slice(dataset, slice_dim='z'):
    def means_ok(means):
        median_of_means = np.median(means)
        if np.any(means < median_of_means*0.5):
            return False
        return True
    
    idx_dim = 'zyx'.find(slice_dim)
    axes_other = tuple(i for i in range(3) if (i != idx_dim))
    assert idx_dim >= 0
    
    if opts.n_max is not None:
        indices = np.round(np.linspace(0, len(dataset) - 1, opts.n_max)).astype(int)
    else:
        indices = range(len(dataset))
    
    report_str = []
    report_str.append('***** REPORT *****')
    n_bad_images = 0
    for i, idx in enumerate(indices):
        fails = []
        ar = dataset.get_item_sel(idx, opts.element_num, apply_transforms=opts.apply_transforms)
        if ar is None:
            continue
        print('DEBUG: shape', ar.shape)
        if ar.shape[0] < 32:
            fails.append('Z-dim size is below minimum. min: {}, got: {}'.format(32, ar.shape[0]))
        n_means = ar.shape[idx_dim]
        slice_vals = range(n_means)
        means = np.mean(ar, axis=axes_other)
        fig, ax = plt.subplots(1)
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
        plt.show()
        plt.close(fig)
        if not means_ok(means):
            fails.append('bad mean pixel intensity for one or more slice')
            n_bad_images += 1
        status = 'pass' if len(fails) == 0 else 'FAIL'
        report_str.append('{} | {}'.format(status, dataset.get_name(idx, opts.element_num)))
        if len(fails) > 0:
            report_str.extend(map(lambda x : '  ' + x, fails))
    report_str.append('{}/{} bad images'.format(n_bad_images, len(indices)))
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
        ar = dataset.get_item_sel(idx, opts.element_num, apply_transforms=opts.apply_transforms)
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

def check_blank_slices(volume, slice_dim='z'):
    idx_dim = 'zyx'.find(slice_dim)
    axes_other = tuple(i for i in range(3) if (i != idx_dim))
    assert idx_dim >= 0
    
    means = np.mean(volume, axis=axes_other)
    assert means.ndim == 1
    threshold = 10
    median_of_means = np.median(means)
    mask_bads = np.logical_or(means < threshold, means < 0.5*median_of_means)
    if np.count_nonzero(mask_bads):
        idx_bads = np.flatnonzero(mask_bads)
        msg = 'bad {:s}: {:s}'.format(slice_dim, str(tuple(idx_bads)))
        return False, msg
    return True, 'okay'

def check_dataset_element(vols):
    if vols is None:
        return False, 'could not create elements'
    titles = ('signal', 'target')
    check_list = [
        check_blank_slices,
    ]
    messages = []
    for i in range(2):
        vol = vols[i]
        for check in check_list:
            vol_passed, msg = check(vol)
            if not vol_passed:
                messages.append(titles[i] + ' ' + msg)
    return len(messages) == 0, ';'.join(messages)
            
def to_uint8(ar, val_min, val_max):
    ar_new = ar.copy()
    if val_min is not None: ar_new[ar_new < val_min] = val_min
    if val_max is not None: ar_new[ar_new > val_max] = val_max
    ar_new -= np.min(ar_new)
    ar_new = ar_new/np.max(ar_new)*256.0
    ar_new[ar_new >= 256.0] = 255.0
    return ar_new.astype(np.uint8)

def check_dataset():
    if os.path.isdir(opts.path_source):
        paths = [i.path for i in os.scandir(opts.path_source) if i.is_file() and i.path.lower().endswith('.czi')]  # order is arbitrary
        dataset = fnet.data.DataSet3(
            czis=paths,
            signal=-1,
            target=2,
            transforms=None
        )
    else:
        dataset = fnet.data.functions.load_dataset(opts.path_source)
        # dataset = fnet.data.DataSet2(
        #     path_load=opts.path_source,
        #     train_select=True
        # )
    print(dataset)
    if opts.plot_sel == 'boxes':
        plot_boxplots(dataset)
    if opts.plot_sel == 'means':
        plot_mean_intensity_per_slice(dataset, slice_dim='z')

def test_check_blank_slices():
    rng = np.random.RandomState(666)
    ar_test = rng.randint(100, 256, size=(10, 8, 12))
    idx_bads = rng.randint(0, ar_test.shape[0], size=3)
    ar_test[idx_bads, :, :] = 0
    result = check_blank_slices(ar_test)
    print(result)
    assert result[0] is False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train_csv', help='path to training set csv')
    parser.add_argument('--path_test_csv', help='path to test set csv')
    parser.add_argument('--path_output', default='data/dataset_eval', help='path to directory for output images and results')
    parser.add_argument('--n_elements', help='maximum number of dataset elements to evaluate')
    opts = parser.parse_args()

    dataset = fnet.data.DataSet(
        path_train_csv = opts.path_train_csv,
        path_test_csv = opts.path_test_csv,
        scale_z = None,
        scale_xy = None,
        transforms = None,
    )
    dataset.use_train_set()
    path_dir_output = os.path.join(opts.path_output, os.path.basename(opts.path_train_csv).split('.')[0])
    if not os.path.exists(path_dir_output):
        os.makedirs(path_dir_output)

    # layout options
    n_z_per_img = 3
    padding_h = 5
    padding_v = 5
    val_range_signal = (-10, 10)
    val_range_target = (-3, 7)
    val_ranges = (val_range_signal, val_range_target)
    pass_fails = []
    for i in range(len(dataset)):
        print('checking element:', i)
        vols = dataset[i]
        if vols is not None:
            print('shapes:', vols[0].shape, vols[1].shape)
            shape = (vols[0].shape[1]*n_z_per_img + padding_v*(n_z_per_img - 1),
                     vols[0].shape[2]*2 + padding_h)
            z_indices = [int((i + 1)*(vols[0].shape[0]/(n_z_per_img + 1))) for i in range(n_z_per_img)]

            img_ex = np.ones(shape, dtype=np.uint8)*255
            for idx_z, z in enumerate(z_indices):
                offset_y = idx_z*(vols[0].shape[1] + padding_v)
                for idx_vol, vol in enumerate(vols):
                    offset_x = idx_vol*(vol.shape[2] + padding_h)
                    vol_uint8 = to_uint8(fnet.data.sub_mean_norm(vol), *val_ranges[idx_vol])
                    img_ex[offset_y:offset_y + vol.shape[1], offset_x:offset_x + vol.shape[2]] = vol_uint8[z, :, :]
            path_save = os.path.join(path_dir_output, 'img_{:02d}.png'.format(i))
            scipy.misc.imsave(path_save, img_ex)
            print('saved image to:', path_save)
        element_passed, msg = check_dataset_element(vols)
        pass_fails.append({
            'path_czi': dataset._df_active['path_czi'].iloc[i],
            'pass': element_passed,
            'reason': msg,
        })
        if opts.n_elements is not None and (i + 1) == opts.n_elements:
            break
    path_csv_out = os.path.join(path_dir_output, 'results.csv')
    df_pass_fails = pd.DataFrame(pass_fails)
    df_pass_fails.to_csv(path_csv_out, index=False)
    print('saved results to:', path_csv_out)
    
    path_csv_rejects = os.path.join(opts.path_output, 'czi_rejects.csv')
    if os.path.exists(path_csv_rejects):
        df_rejects = pd.read_csv(path_csv_rejects)
        df_rejects = pd.concat(df_rejects, df_pass_fails[df_pass_fails['pass'] == False], ignore_index=True)
    else:
        df_rejects = df_pass_fails[df_pass_fails['pass'] == False]
    df_rejects.to_csv(path_csv_rejects, index=False)
    print('saved results to:', path_csv_rejects)

if __name__ == '__main__':
    # test_check_blank_slices()
    main()
