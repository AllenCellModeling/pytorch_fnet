"""Blend fluorescence images together."""

import matplotlib
import argparse
import os
import pdb
import tifffile
import numpy as np
import re
import pdb

def convert_ar_float_to_uint8(ar, val_range=None):
    val_min, val_max = (None, None) if val_range is None else val_range
    ar_new = ar.astype(np.float)
    if val_min is not None:
        ar_new[ar_new < val_min] = val_min
    if val_max is not None:
        ar_new[ar_new > val_max] = val_max
    ar_new -= np.min(ar_new) if val_min is None else val_min
    ar_new /= np.max(ar_new) if val_max is None else (val_max - val_min)
    ar_new *= 256.0
    ar_new[ar_new >= 256.0] = 255.0
    return ar_new.astype(np.uint8)

def convert_ar_grayscale_to_rgb(ar, hue_sat, val_range=None):
    """Converts grayscale image to RGB uint8 image.
    ar - numpy.ndarray representing grayscale image
    hue_sat - 2-element tuple representing a color's hue and saturation values.
              Elements should be [0.0, 1.0] if floats, [0, 255] if ints.
    """
    hue = hue_sat[0]/256.0 if isinstance(hue_sat[0], int) else hue_sat[0]
    sat = hue_sat[1]/256.0 if isinstance(hue_sat[1], int) else hue_sat[1]

    ar_float = ar.astype(np.float)

    val_min, val_max = (None, None) if val_range is None else val_range
    if val_min is not None:
        ar_float[ar_float < val_min] = val_min
    if val_max is not None:
        ar_float[ar_float > val_max] = val_max

    ar_float -= np.min(ar_float) if val_min is None else val_min
    ar_float /= np.max(ar_float) if val_max is None else (val_max - val_min)
    
    ar_hsv = np.zeros(ar.shape + (3, ), dtype=np.float)
    ar_hsv[..., 0] = hue
    ar_hsv[..., 1] = sat
    ar_hsv[..., 2] = ar_float
    ar_rgb = matplotlib.colors.hsv_to_rgb(ar_hsv)
    ar_rgb *= 256.0
    ar_rgb[ar_rgb == 256.0] = 255.0
    return ar_rgb.astype(np.uint8)

def blend_ar(ars, weights):
    shape_exp = ars[0].shape
    ar_all = np.zeros(ars[0].shape, dtype=np.float)
    for i in range(len(ars)):
        assert weights[i] >= 0.0
        assert ars[i].shape == shape_exp
        ar_all += weights[i]*ars[i]
    ar_all -= np.min(ar_all)
    ar_all /= np.max(ar_all)
    ar_all *= 256.0
    ar_all[ar_all == 256.0] = 255.0
    return ar_all.astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dir', help='path ')
    parser.add_argument('--timelapse', action='store_true', help='set if images are part of a timelapse to use fixed contrast adjustment')
    opts = parser.parse_args()
    
    assert os.path.exists(opts.path_dir)
    if not opts.timelapse:
        raise NotImplementedError
    
    print('source directory:', opts.path_dir)
    
    path_out_dir = opts.path_dir
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)
    paths_sources = [os.path.join(opts.path_dir, i) for i in os.listdir(opts.path_dir) if i.lower().endswith('.tif')]
    paths_sources.sort()
    
    imgs_rgb = []
    last_pretag = None
    pattern = re.compile(r'(.+_\d+)_')
    path_combo_basename = ''
    for i, path in enumerate(paths_sources):
        path_source_basename = os.path.basename(path)
        match = pattern.search(path_source_basename)
        make_blend = False
        if match:
            pretag = match.group(1)
            if last_pretag != pretag:
                if last_pretag is not None:
                    path_combo_basename = last_pretag + '_combo.tif'
                    make_blend = True
                last_pretag = pretag
        else:
            # handle case when there is no number in file name
            raise NotImplementedError

        if make_blend:
            print(len(imgs_rgb), path_combo_basename)
            img_combo = blend_ar(imgs_rgb, (1/len(imgs_rgb),)*len(imgs_rgb))    
            path_img_combo = os.path.join(path_out_dir, path_combo_basename)
            tifffile.imsave(path_img_combo, img_combo, photometric='rgb')
            print('saved:', path_img_combo)
            imgs_rgb = []
        
        img_source = tifffile.imread(path)
        imgs_rgb.append(img_source)

    path_combo_basename = last_pretag + '_combo.tif'
    print(len(imgs_rgb), path_combo_basename)
    img_combo = blend_ar(imgs_rgb, (1/len(imgs_rgb),)*len(imgs_rgb))    
    path_img_combo = os.path.join(path_out_dir, path_combo_basename)
    tifffile.imsave(path_img_combo, img_combo, photometric='rgb')
    print('saved:', path_img_combo)
