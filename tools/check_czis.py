import sys
sys.path.append('.')
import os
import argparse
import pandas as pd
import numpy as np
import scipy.misc
from fnet.data.czireader import CziReader
import fnet.data
import time
import functions
import pdb

def to_uint8(ar, val_min, val_max):
    ar_new = ar.copy()
    if val_min is not None: ar_new[ar_new < val_min] = val_min
    if val_max is not None: ar_new[ar_new > val_max] = val_max
    ar_new -= np.min(ar_new)
    ar_new = ar_new/np.max(ar_new)*256.0
    ar_new[ar_new >= 256.0] = 255.0
    return ar_new.astype(np.uint8)

def save_vol_slices(path_save, vols):
    if len(vols) > 2:
        print('> 2 volumes not yet supported')
        return
    # layout options
    n_z_per_img = 3
    padding_h = 5
    padding_v = 5
    val_range_signal = (-10, 10)
    val_range_target = (-3, 7)
    val_ranges = (val_range_signal, ) + (val_range_target, )*(len(vols) - 1)
    if vols is not None:
        # print('shapes:', vols[0].shape, vols[1].shape)
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
        scipy.misc.imsave(path_save, img_ex)
        print('saved image to:', path_save)
        
def check_blank_slices(volume, slice_dim='z'):
    idx_dim = 'zyx'.find(slice_dim)
    axes_other = tuple(i for i in range(3) if (i != idx_dim))
    assert idx_dim >= 0
    means = np.mean(volume, axis=axes_other)
    assert means.ndim == 1
    threshold = 20
    median_of_means = np.median(means)
    # mask_bads = np.logical_or(means < threshold, means < 0.5*median_of_means)
    mask_bads = means < threshold
    if np.count_nonzero(mask_bads):
        idx_bads = np.flatnonzero(mask_bads)
        msg = 'bad {:s}: {:s}'.format(slice_dim, str(tuple(idx_bads)))
        return msg
    return ''

def eval_czi(path_czi, channels_sel, path_save=None):
    """
    path_czi : path to CZI file
    channels_sel : list of channels to check
    """
    if not os.path.exists(path_czi):
        return 'file does not exist'
    vol_check_list = [
        check_blank_slices,
    ]
    print('reading:', path_czi)
    czi = CziReader(path_czi)
    if channels_sel == -1:
        channels_sel = range(czi.get_size('C'))
    messages = []
    vols = []
    for chan in channels_sel:
        vol = czi.get_volume(chan)
        vols.append(vol)
        for check in vol_check_list:
            msg = check(vol)
            if msg != '':
                messages.append('chan {:d} {:s}'.format(chan, msg))
    if path_save is not None:
        save_vol_slices(path_save, vols)
    if len(messages) > 0:
        return ';'.join(messages)
    return ''

def get_df_check(df_source, target):
    """Filter source DataFrame based on target."""
    if target is None:
        return df_source
    if target in [
            'dna',
            'membrane',
            'dic-lamin_b1',
            'dic-membrane'
    ]:
        raise NotImplementedError
    else:
        df_check = df_source[df_source['structureProteinName'] == target].copy()
        print(df_check.shape)
    return df_check

def get_column_names(target):
    """Determine what column names to look for depending on target."""
    if target is None:
        return []
    if target in ['dna', 'membrane']:
        raise NotImplementedError
    return ['lightChannel', 'structureChannel']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_all_channels', action='store_true',  help='set to check all channels of czi')
    parser.add_argument('--n_files', type=int, help='maximum number of CZI files to evaluate')
    parser.add_argument('--path_csv', help='path to csv with CZI paths')
    parser.add_argument('--path_output_dir', default='data/czi_eval', help='path to directory for output images and results')
    parser.add_argument('--save_layouts', action='store_true', help='save slice layoust for each element')
    parser.add_argument('--shuffle', action='store_true', help='shuffle czis in csv')
    opts = parser.parse_args()
    
    time_start = time.time()
    interval_checkpoint = 500
    
    df_source = pd.read_csv(opts.path_csv)
    print('read csv:', opts.path_csv)
    basename = os.path.basename(opts.path_csv)

    if not os.path.exists(opts.path_output_dir):
        os.makedirs(opts.path_output_dir)

    n_files = opts.n_files if opts.n_files is not None else df_source.shape[0]
    if opts.shuffle:
        rng = np.random.RandomState(opts.seed)
        indices = rng.choice(np.arange(df_source.shape[0]), replace=False, size=n_files)
    else:
        indices = range(n_files)
    check_all_channels = False
    if (opts.check_all_channels or
        'channel_signal' not in df_source.columns or
        'channel_target' not in df_source.columns):
        check_all_channels = True
    entries = []
    path_save_csv = os.path.join(
        opts.path_output_dir,
        basename + '_check.csv'
    )
    count = 0
    for idx in indices:
        count += 1
        print('checking: idx {:d} | count: {:d} of {:d}'.format(idx, count, len(indices)))
        channels_sel = []
        entry = {}
        czi_row = df_source.iloc[idx].copy()
        path_czi = czi_row['path_czi']
        if check_all_channels:
            channels_check = -1
        else:
            channels_check = [czi_row['channel_signal'], czi_row['channel_target']]
        path_save = os.path.join(opts.path_output_dir, '{:s}_element_{:03d}.png'.format(basename, idx)) if opts.save_layouts else None
        reason = eval_czi(path_czi, channels_check, path_save)
        czi_row['pass'] = reason == ''
        czi_row['reason'] = reason
        entries.append(czi_row)
        if (count % interval_checkpoint == 0) or (count >= len(indices)):
            functions.save_backup(path_save_csv)
            pd.DataFrame(entries).to_csv(path_save_csv, index=False)
            print('saved results csv to:', path_save_csv)
            print('time elapsed: {:.1f}'.format(time.time() - time_start))

if __name__ == '__main__':
    main()
