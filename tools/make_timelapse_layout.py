import argparse
import functions
import json
import tifffile
import scipy
import numpy as np
import pdb
import re
import os

def get_file_paths(paths_sources, tags):
    pattern = re.compile(r'(\d+)_')
    file_records = {}  # keys wille be file indexes
    for path_source in paths_sources:
        paths_files = [i.path for i in os.scandir(path_source) if i.is_file()]
        for path_file in paths_files:
            basename = os.path.basename(path_file)
            match = pattern.search(basename)
            if match is not None:
                idx = int(match.groups()[0])
                tag_path_dict = file_records.get(idx)
                if tag_path_dict is None:
                    tag_path_dict = {}
                for tag in tags:
                    if tag in basename:
                        tag_path_dict[tag] = path_file
                        file_records[idx] = tag_path_dict
    return file_records

def get_one_layout_per_frame(
        file_records,
        frames_layout,
        tags,
        z_slice,
        path_save_dir,
):
    for frame in frames_layout:
        img_frame = get_img_layout(
            file_records, [frame], tags,
            z_slice = z_slice,
            color = 0,
        )
        path_save_img_frame = os.path.join(path_save_dir, '{:03d}_timelapse_layout_z{:02d}.png'.format(frame, z_slice))
        scipy.misc.imsave(path_save_img_frame, img_frame)
        print('saved image to:', path_save_img_frame)

def get_img_layout(
        file_records, frames_layout, tags,
        z_slice = 15,
        spacing_v = 5,
        spacing_h = 5,
        color=255,
):
    img_layout = None
    n_rows = len(frames_layout)
    n_cols = len(tags)
    for r, idx_frame in enumerate(frames_layout):
        print('frame:', idx_frame)
        tag_path_dict = file_records.get(idx_frame)
        if tag_path_dict is None:
            print('frame {:d} did not exist. skipping....'.format(idx_frame))
            continue
        offset_r = None
        for c, tag in enumerate(tags):
            path_this_tag = tag_path_dict.get(tag)
            print('using path:', path_this_tag)
            if path_this_tag is None:
                continue
            ar = tifffile.imread(path_this_tag)
            if img_layout is None:
                shape_layout = (
                    ar.shape[1]*n_rows + (n_rows - 1)*spacing_v,
                    ar.shape[2]*n_cols + (n_cols - 1)*spacing_h,
                    3,  # assume color image
                )
                img_layout = np.ones(shape_layout, dtype=np.uint8)*color
            if offset_r is None:
                offset_r = r*(ar.shape[1] + spacing_v)
            offset_c = c*(ar.shape[2] + spacing_h)
            img = np.atleast_3d(ar[z_slice, :, :, ])
            if (r, c) == (0, 0):
                functions.add_scale_bar(img, 20, 0.3)
            img_layout[offset_r:offset_r + ar.shape[1], offset_c:offset_c + ar.shape[2], ] = img
    return img_layout
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_source_dir', nargs='+', help='directory of prediction tifs')
    parser.add_argument('--tags', nargs='+', help='tags to identify file content')
    parser.add_argument('--path_save_dir', default='fnet_paper/timelapse', help='directory to save results')
    parser.add_argument('--interval_frames', type=int, default=5, help='number of frames between layout timepoints')
    parser.add_argument('--z_slice', type=int, default=15, help='z-slice selection')
    parser.add_argument('--per_frame', action='store_true', help='set to output 1 layout per frame')
    opts = parser.parse_args()

    file_records = get_file_paths(opts.path_source_dir, opts.tags)
    frame_list = sorted(file_records.keys())
    frame_min, frame_max = min(frame_list), max(frame_list)
    frames_layout = range(frame_min, frame_max + 1, opts.interval_frames if not opts.per_frame else 1)
    print('frames:', list(frames_layout))
    if not os.path.exists(opts.path_save_dir):
        os.makedirs(opts.path_save_dir)
    dict_log = vars(opts)
    dict_log['frames'] = list(frames_layout)
    path_info_layout = os.path.join(opts.path_save_dir, 'layout_info.json')
    with open(path_info_layout, 'w') as fo:
        json.dump(dict_log, fo, indent=4)
        print('saved options to:', path_info_layout)
    
    if opts.per_frame:
        get_one_layout_per_frame(
            file_records,
            frames_layout,
            opts.tags,
            opts.z_slice,
            opts.path_save_dir,
        )
    else:
        img_layout = get_img_layout(
            file_records, frames_layout, opts.tags,
            z_slice = opts.z_slice,
        )
        path_img_layout = os.path.join(opts.path_save_dir, 'timelapse_layout_z{:02d}.png'.format(opts.z_slice))
        scipy.misc.imsave(path_img_layout, img_layout)
    print('Done!')
