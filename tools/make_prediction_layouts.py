import argparse
import numpy as np
import os
import re
import tifffile
import scipy.misc
import pdb

TAGS_SIGNAL = ['signal', 'bright']
TAGS_TARGET = ['target', 'DNA']
TAGS_PREDICTION = ['prediction']

# tag to contrast adjustment map
ADJUSTMENT_MAP = {
    'fibrillarin': {
        'target': (-0.5, 12),
        'prediction': (-0.5, 12),
    }
}

def to_uint8(ar, val_min, val_max):
    ar_new = ar.copy()
    if val_min is not None: ar_new[ar_new < val_min] = val_min
    if val_max is not None: ar_new[ar_new > val_max] = val_max
    ar_new -= np.min(ar_new)
    ar_new = ar_new/np.max(ar_new)*256.0
    ar_new[ar_new >= 256.0] = 255.0
    return ar_new.astype(np.uint8)

def add_scale_bar(img, width, um_per_pixel, color=255, thickness=5, vpad=10, hpad=10):
    """Annotate image with scale bar.

    img - np.array(dytpe=np.uint8)
    width - bar width in microns
    px_per_um - float
    """
    width_in_pixels = int(np.round(width/um_per_pixel))
    offset_y = img.shape[0] - thickness - vpad
    offset_x = img.shape[1] - width_in_pixels - hpad
    img[offset_y:offset_y + thickness, offset_x:offset_x + width_in_pixels] = color
    return img

def find_source_dirs(path_root_dir):
    """Find source directories to make layouts, going at most 1 layer deep.

    Returns : list of source directories
    """
    def is_source_dir(path):
        if not os.path.isdir(path):
            return False
        has_signal, has_target, has_prediction = False, False, False
        for entry in [i.path for i in os.scandir(path) if i.is_file()]:
            if any(tag in entry for tag in TAGS_SIGNAL):
                has_signal = True
            if any(tag in entry for tag in TAGS_TARGET):
                has_target = True
            if any(tag in entry for tag in TAGS_PREDICTION):
                has_prediction = True
        return has_signal and has_target and has_prediction
    
    if is_source_dir(path_root_dir):
        return [path_root_dir]
    results = []
    for entry in os.scandir(path_root_dir):
        if is_source_dir(entry.path):
            results.append(entry.path)
    return results

def make_layout_image(
        path_source_dir,
        path_save_dir,
        n_images = 10,
        adjustment_map = {},
):
    # layout options
    n_z_per_img = 5
    padding_h = 5
    padding_v = 5
    shape_z = 32

    val_range_signal = adjustment_map.get('signal', (-10, 10))
    val_range_target = adjustment_map.get('target', (-3, 7))
    val_range_prediction = adjustment_map.get('prediction', (-0.9, 6))
        
    paths_tifs = sorted([i.path for i in os.scandir(path_source_dir) if i.is_file() and i.path.lower().endswith('.tif')])
    pattern = re.compile(r'.+_test_(\d+)_')

    # z_indices = [int((i + 1)*(shape_z/(n_z_per_img + 1))) for i in range(n_z_per_img)]
    inc = int(shape_z/(n_z_per_img + 1))
    z_indices = list(range(inc, shape_z, inc))[:n_z_per_img]

    z_indices = np.flip(z_indices, axis=0)
    print('z_indices:', z_indices)

    idx_old = None
    count_images = 0
    for i, path in enumerate(paths_tifs):
        idx_col = None
        path_basename = os.path.basename(path)
        match = pattern.search(path_basename)
        if match is None:
            break
        idx_img = match.groups()[0]
        if any(tag in path_basename for tag in TAGS_SIGNAL):
            idx_col = 0
            val_min, val_max = val_range_signal
        if any(tag in path_basename for tag in TAGS_TARGET):
            idx_col = 1
            val_min, val_max = val_range_target
        if any(tag in path_basename for tag in TAGS_PREDICTION):
            idx_col = 2
            val_min, val_max = val_range_prediction
        if idx_col is not None:
            ar_pre = tifffile.imread(path)
            print('DEBUG: {:30s} {:6.2f} {:6.2f}'.format(path_basename, np.min(ar_pre), np.max(ar_pre)))
            ar = to_uint8(ar_pre, val_min, val_max)       
            if idx_img != idx_old:
                n_cols_done = 0
                idx_old = idx_img
                shape = (ar.shape[1]*n_z_per_img + (n_z_per_img - 1)*padding_h, ar.shape[2]*3 + 2*padding_v)
                ar_fig = np.ones(shape, dtype=np.uint8)*255
            offset_x = idx_col*(ar.shape[2] + padding_h)
            for idx_row, z_index in enumerate(z_indices):
                offset_y = idx_row*(ar.shape[1] + padding_v)
                img = ar[z_index, ].copy()
                if (idx_col, idx_row) == (0, 0):
                    add_scale_bar(img, 20, 0.3)
                ar_fig[offset_y:offset_y + ar.shape[1], offset_x:offset_x + ar.shape[2]] = img
            n_cols_done += 1
            if n_cols_done == 3:
                name_set = os.path.basename(os.path.dirname(path))
                path_save_base = os.path.join(
                    path_save_dir,
                    name_set,
                )
                path_save_img = os.path.join(
                    path_save_base,
                    match.group() + 'layout.png',
                )
                if not os.path.exists(path_save_base):
                    os.makedirs(path_save_base)
                scipy.misc.imsave(path_save_img, ar_fig)
                print('saved image to:', path_save_img)
                count_images += 1
        if count_images >= n_images:
            break
    path_save_log = os.path.join(
        path_save_base,
        'z_slices.txt',
    )
    with open(path_save_log, 'w') as fo:
        print('z_slices:', z_indices, file=fo)
        print('saved z_slices info to:', path_save_log)

def get_adjustment_map(path):
    path_basename = os.path.basename(path)
    for k in ADJUSTMENT_MAP:
        if k in path_basename:
            return ADJUSTMENT_MAP.get(k)
    return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_source_dir', help='directory (or directory of directory) of prediction results')
    parser.add_argument('--path_save_dir', default='fnet_paper/example_predictions', help='directory to save results')
    parser.add_argument('--n_images', type=int, default=8, help='number of examples to lay out')
    opts = parser.parse_args()
    
    for path_source_dir in find_source_dirs(opts.path_source_dir):
        print(path_source_dir)
        adjustment_map = get_adjustment_map(path_source_dir)
        make_layout_image(
            path_source_dir = path_source_dir,
            path_save_dir = opts.path_save_dir,
            n_images = opts.n_images,
            adjustment_map = adjustment_map,
        )

