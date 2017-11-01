"""Blend fluorescence images together."""

import argparse
import os
import pdb
import tifffile
import numpy as np
import re
import pdb
import sys

def blend_ar_0(ars, weights):
    shape_exp = ars[0].shape
    ar_all = np.zeros(shape_exp, dtype=np.float)
    for i in range(len(ars)):
        assert weights[i] >= 0.0
        assert ars[i].shape == shape_exp
        ar_all += weights[i]*ars[i]
    ar_all -= np.min(ar_all)
    ar_all /= np.max(ar_all)
    ar_all *= 256.0
    ar_all[ar_all >= 256.0] = 255.0
    return ar_all.astype(np.uint8)

def blend_ar_1(ars, weights):
    # don't use weights
    shape_exp = ars[0].shape
    ar_all = np.zeros(shape_exp, dtype=np.float)
    for i in range(len(ars)):
        ar_all += ars[i]
    ar_all -= np.min(ar_all)
    ar_all /= np.max(ar_all)
    ar_all *= 256.0
    ar_all[ar_all >= 256.0] = 255.0
    return ar_all.astype(np.uint8)

def make_example_tiff():
    hue_sats = palette_seaborn_colorblind[3:]
    img_r_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    img_g_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    img_b_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    for z in range(img_r_pre.shape[0]):
        img_r_pre[z, :20, :50] = 255 - 2*z
        img_g_pre[z, 40:60, 50:100] = 100 + 2*z
        img_b_pre[z, 65:85, 150:200] = 150 + 2*z
    path_base = 'test_output'
    img_r = convert_ar_grayscale_to_rgb(img_r_pre, hue_sats[0])
    img_g = convert_ar_grayscale_to_rgb(img_g_pre, hue_sats[1])
    img_b = convert_ar_grayscale_to_rgb(img_b_pre, hue_sats[2])
    tifffile.imsave(os.path.join(path_base, 'r.tif'), img_r, photometric='rgb')
    tifffile.imsave(os.path.join(path_base, 'g.tif'), img_g, photometric='rgb')
    tifffile.imsave(os.path.join(path_base, 'b.tif'), img_b, photometric='rgb')
    img_all = blend_ar((img_r, img_g, img_b), (1/3, 1/3, 1/3))
    tifffile.imsave(os.path.join(path_base, 'all.tif'), img_all, photometric='rgb')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_source_dir', help='path to source directory')
    parser.add_argument('--path_output_dir', help='path to output directory')
    parser.add_argument('--tags', nargs='+', default=['dna', 'fibrillarin', 'lamin_b1', 'sec61', 'tom20'], help='tags of files to blend')
    parser.add_argument('--blender', default='1',  help='tags of files to blend')
    opts = parser.parse_args()
    
    assert os.path.exists(opts.path_source_dir)
    print('source directory:', opts.path_source_dir)

    str_blender = 'blend_ar_' + opts.blender
    blend_ar = getattr(sys.modules[__name__], str_blender)
    
    if not os.path.exists(opts.path_output_dir):
        os.makedirs(opts.path_output_dir)
    paths_sources = [i.path for i in os.scandir(opts.path_source_dir) if i.path.lower().endswith('.tif')]
    paths_sources.sort()

    n_combined = len(opts.tags)
    imgs_blend = []
    pattern = re.compile(r'(\d+)_')
    tag_output = '{:d}_channel'.format(n_combined)
    idx_set_old = None
    for i, path in enumerate(paths_sources):
        path_basename = os.path.basename(path)
        match = pattern.search(path_basename)
        if match is None:
            print(path)
            raise NotImplementedError
        idx_set = int(match.groups()[0])
        if i == 0:
            idx_set_old = idx_set
        if any(tag in path_basename for tag in opts.tags):
            print('reading:', path)
            imgs_blend.append(tifffile.imread(path))
        if len(imgs_blend) == n_combined:
            assert idx_set_old == idx_set
            path_output = os.path.join(opts.path_output_dir, '{:03d}_{:s}.tif'.format(idx_set, tag_output))
            img_combo = blend_ar(imgs_blend, (1/len(imgs_blend),)*len(imgs_blend))    
            tifffile.imsave(path_output, img_combo, photometric='rgb')
            print('saved:', path_output)
            imgs_blend = []
        idx_set_old = idx_set
