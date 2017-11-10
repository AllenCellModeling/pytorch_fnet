import numpy as np
import os
import re
import tifffile
import scipy.misc
import shutil
import pdb

def save_backup(path_file, n_backups=5):
    if not os.path.exists(path_file):
        return
    path_dir, path_base = os.path.split(path_file)
    path_backup_dir = os.path.join(path_dir, 'backups')
    if not os.path.exists(path_backup_dir):
        os.makedirs(path_backup_dir)
    paths_existing_backups = [i.path for i in os.scandir(path_backup_dir)
                              if (path_base in i.path and i.path.split('.')[-1].isdigit())]
    paths_existing_backups.sort(key=lambda x: os.path.getmtime(x))
    tag = 0
    if len(paths_existing_backups) > 0:
        tag = (int(paths_existing_backups[-1].split('.')[-1]) + 1) % 100
    paths_delete = paths_existing_backups[:-(n_backups - 1)] if n_backups > 1 else paths_existing_backups
    for path in paths_delete:
        os.remove(path)
    path_backup = os.path.join(path_backup_dir, path_base + '.{:02}'.format(tag))
    shutil.copyfile(path_file, path_backup)
    print('wrote to:', path_backup)
    
def to_uint8(ar, val_min, val_max):
    ar_new = ar.copy()
    if val_min is not None: ar_new[ar_new < val_min] = val_min
    if val_max is not None: ar_new[ar_new > val_max] = val_max
    ar_new -= np.min(ar_new)
    ar_new = ar_new/np.max(ar_new)*256.0
    ar_new[ar_new >= 256.0] = 255.0
    return ar_new.astype(np.uint8)

def add_scale_bar(img, width, um_per_pixel, color=255, thickness=5, pad_v=10, pad_h=10):
    """Annotate image with scale bar.

    img - np.array(dytpe=np.uint8)
    width - bar width in microns
    px_per_um - float
    """
    width_in_pixels = int(np.round(width/um_per_pixel))
    offset_y = img.shape[0] - thickness - pad_v
    offset_x = img.shape[1] - width_in_pixels - pad_h
    img[offset_y:offset_y + thickness, offset_x:offset_x + width_in_pixels] = color
    return img

