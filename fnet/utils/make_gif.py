import numpy as np
import os
import pandas as pd
import subprocess
import tifffile
import pdb


def to_uint8(ar, range_val=None):
    if range_val is None:
        covfefe
        range_val = (np.min(ar), np.max(ar))
    ar_new = ar.copy()
    ar_new[ar_new < range_val[0]] = range_val[0]
    ar_new[ar_new > range_val[1]] = range_val[1]
    ar_new = (ar_new - range_val[0])/(range_val[1] - range_val[0])*256.0
    ar_new[ar_new >= 256.0] = 255.0
    return ar_new.astype(np.uint8)

def make_timelapse_gif(
        path_source_csv,
        col,
        z_slice,
        path_out_dir,
        tag = '',
        range_percentile = (0.5, 99.5),
):
    df = pd.read_csv(path_source_csv)
    print(df.shape, df.columns)
    # sample from up to 8 images
    n_samples = min(8, df.shape[0])
    indices_sample = np.random.choice(range(df.shape[0]), size=n_samples, replace=False)
    dirname = os.path.dirname(path_source_csv)
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)
    print('DEBUG: col', col)
    idx_col = df.columns.get_loc(col)
    imgs = []
    for idx in indices_sample:
        path_img = os.path.join(dirname, df.iloc[idx, idx_col])
        print(path_img)
        imgs.append(tifffile.imread(path_img))
        print(np.percentile(imgs[-1], range_percentile))
    range_val = np.percentile(imgs, range_percentile)
    print('DEBUG: range_val:', range_val)

    imgs_out = []
    for idx in range(df.shape[0]):
        path_img = os.path.join(dirname, df.iloc[idx, idx_col])
        img = tifffile.imread(path_img)[:, z_slice, ]
        img_uint8 = to_uint8(img, range_val)
        path_save = os.path.join(
            path_out_dir,
            '{:s}_z{:02d}_{:03d}.tiff'.format(tag, z_slice, idx)
        )
        tifffile.imsave(path_save, img_uint8)
        print('wrote:', path_save)
        imgs_out.append(path_save)
    path_gif = os.path.join(path_out_dir, '{:s}_z{:02d}.gif'.format(tag, z_slice))
    cmd = 'convert -delay {:d} {:s} {:s}'.format(15, ' '.join(imgs_out), path_gif)
    subprocess.run(cmd, shell=True, check=True)
    print('wrote:', path_gif)
    
if __name__ == '__main__':
    make_timelapse_gif(
        path_source_csv = 'results/timelapse/timelapse_wt/predictions.csv',
        col = 'path_prediction_tom20',
        z_slice = 32,
        path_out_dir = 'animated/timelapse_wt/tom20',
        tag = 'bf',
        range_percentile = (0.1, 99.9),
    )
