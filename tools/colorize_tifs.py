"""Coverts grayscale TIFF images in directory to RGB TIFFs and produce a combined image."""

import matplotlib
import argparse
import os
import pdb
import tifffile
import numpy as np
import pdb

palette_seaborn_colorblind = ((142, 255),
                              (115, 255),
                              (18, 255),
                              (231, 103),
                              (39, 184),
                              (142, 160),
)

tags_to_colors_map = {
    'dna': palette_seaborn_colorblind[0],
    'fibrillarin': palette_seaborn_colorblind[1],
    'lamin_b1': palette_seaborn_colorblind[2],
    'sec61': palette_seaborn_colorblind[3],
    'tom20': palette_seaborn_colorblind[4],
    'signal': None,
    'target': None,
}

tags_to_val_ranges_map = {
    'dna': (-0.7, 5.0),
    'fibrillarin': (-0.45, 14.0),
    'lamin_b1': (-0.85, 14.0),
    'sec61': (-1.0, 5.0),
    'tom20': (-0.67, 10.0),
    'signal': (-8.5, 12),
    'target': (-5.0, 13.0),
}

def convert_ar_float_to_uint8(ar, val_range=None):
    ar_new = ar.astype(np.float)
    val_min, val_max = (None, None) if val_range is None else val_range
    if val_min is None:
        val_min = np.min(ar)
    if val_max is None:
        val_max = np.max(ar)
    ar_new[ar_new < val_min] = val_min
    ar_new[ar_new > val_max] = val_max
    val_diff = val_max - val_min
    ar_new -= val_min
    ar_new /= val_diff
    
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

    ar_new = ar.astype(np.float)
    val_min, val_max = (None, None) if val_range is None else val_range
    if val_min is None:
        val_min = np.min(ar)
    if val_max is None:
        val_max = np.max(ar)
    ar_new[ar_new < val_min] = val_min
    ar_new[ar_new > val_max] = val_max
    val_diff = val_max - val_min
    ar_new -= val_min
    ar_new /= val_diff
    
    ar_hsv = np.zeros(ar.shape + (3, ), dtype=np.float)
    ar_hsv[..., 0] = hue
    ar_hsv[..., 1] = sat
    ar_hsv[..., 2] = ar_new
    ar_rgb = matplotlib.colors.hsv_to_rgb(ar_hsv)
    ar_rgb *= 256.0
    ar_rgb[ar_rgb >= 256.0] = 255.0
    return ar_rgb.astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_source_dir', help='path to directory of grayscale tifs')
    parser.add_argument('--tags', nargs='+', help='path to directory of grayscale tifs')
    opts = parser.parse_args()
    
    assert os.path.exists(opts.path_source_dir)
    print('source directory:', opts.path_source_dir)
    path_out_dir = os.path.join(opts.path_source_dir, 'colorize')
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)

    paths_sources = [os.path.join(opts.path_source_dir, i) for i in os.listdir(opts.path_source_dir) if i.lower().endswith('.tif')]
    paths_sources.sort()

    for path in paths_sources:
        path_source_basename = os.path.basename(path)
        if opts.tags is not None and not any(tag in path for tag in opts.tags):
            continue
        hue_sat = None
        for tag in tags_to_colors_map:
            if tag in path_source_basename:
                hue_sat = tags_to_colors_map[tag]
                val_range = tags_to_val_ranges_map[tag]
        img_source = tifffile.imread(path)
        print('DEBUG:', np.min(img_source), np.max(img_source), path)

        if hue_sat is None:
            path_out_basename = path_source_basename.split('.')[0] + '_adjusted.tif'
            img_new = convert_ar_float_to_uint8(img_source, val_range)
        else:
            if 'gray' in path_source_basename:
                path_out_basename = path_source_basename.replace('gray', 'rgb')
            else:
                path_out_basename = path_source_basename.split('.')[0] + '_rgb.tif'
            img_new = convert_ar_grayscale_to_rgb(img_source, hue_sat, val_range)
        path_save = os.path.join(path_out_dir, path_out_basename)
        tifffile.imsave(path_save, img_new, photometric=('minisblack' if (hue_sat is None) else 'rgb'))
        print('saved:', path_save)
