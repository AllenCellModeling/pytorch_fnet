import argparse
import importlib
import pandas as pd
import numpy as np
import tifffile
import fnet.data
import model_modules.fnet_model as model_module
import time
import os
import pdb

def get_df_models(path_source):
    def get_model_info_from_dir(path_dir):
        model_info = {}
        for entry in os.scandir(path_dir):
            if entry.path.lower().endswith('model.p'):
                model_info['path_model'] = entry.path
            elif entry.path.lower().endswith('ds.json'):
                model_info['path_dataset'] = entry.path
        if len(model_info) == 2:
            model_info['name_model'] = os.path.basename(path_dir)
            return model_info
        return None
    
    if path_source.lower().endswith('.csv'):
        return pd.read_csv(path_source)
    
    assert os.path.isdir(path_source)
    model_info = get_model_info_from_dir(path_source)
    if model_info is not None:
        return pd.DataFrame([model_info])
    models = []
    for entry in os.scandir(path_source):
        if os.path.isdir(entry.path):
            model_info = get_model_info_from_dir(entry.path)
            if model_info is not None:
                models.append(model_info)
    return pd.DataFrame(models)

def get_crops_recurse(bounds):
    if len(bounds) <= 1:
        return [[item] for item in bounds[0]]
    crops_rest = get_crops_recurse(bounds[1:])
    return [([item] + item_rest) for item in bounds[0] for item_rest in crops_rest]
        
def calc_bounds(shape_source, shape_sub, overlaps):
    n_dims = len(shape_source)
    bounds = []
    for i in range(n_dims):
        print('i: {:d} | source: {:4d} | sub: {:4d} | overlap: {:4d}'.format(
            i,
            shape_source[i],
            shape_sub[i],
            overlaps[i],
        ))
        n_parts_this_dim = int(np.ceil(shape_source[i]/(shape_sub[i] - overlaps[i])))
        bounds_this_dim = []
        for j in range(n_parts_this_dim):
            start = j*(shape_sub[i] - overlaps[i])
            end = start + shape_sub[i]
            if end > shape_source[i]:
                end = shape_source[i]
                start = end - shape_sub[i]
            bounds_this_dim.append((start, end))
        bounds.append(bounds_this_dim)
    return bounds

def get_batch_prediction(model, batch_x, **kwargs):
    def get_shape_sub(shape_source, pixels_max, multiple_of):
        # find sub volume shape will total pixels smaller pixels_max and where each dim is
        # smaller than the corresponding shape_source dim
        shape_sub = [32]*len(shape_source)
        dims_consider = set(range(len(shape_source)))
        done = False
        print('DEBUG: start shape_sub', shape_sub)
        while not done:
            done = True
            dims_consider_new = set(dims_consider)
            for dim_consider in dims_consider_new:
                shape_new = [(shape_sub[i] + multiple_of) if i == dim_consider else shape_sub[i] for i in range(len(shape_sub))]
                if (shape_new[dim_consider] <= shape_source[dim_consider]) and (np.prod(shape_new) <= pixels_max):
                    done = False
                    shape_sub = shape_new
                else:
                    dims_consider.remove(dim_consider)
        print('DEBUG: final shape_sub', shape_sub)
        return shape_sub

    if model is None:
        return np.zeros(batch_x.shape, batch_x.dtype)
    multiple_of = kwargs.get('multiple_of', 16)
    pixels_max = kwargs.get('pixels_max', 9732096)
    overlaps = kwargs.get('overlaps', [0, 0, 0])
    path_save_partials_dir = kwargs.get('path_save_partials_dir', 'partials')
    shape_source = batch_x.shape[2:]
    shape_sub = get_shape_sub(shape_source, pixels_max, multiple_of)
    print('shape_source:', shape_source)
    print('shape_sub:', shape_sub)
    if not os.path.exists(path_save_partials_dir):
        os.makedirs(path_save_partials_dir)
    bounds = calc_bounds(
        shape_source,
        shape_sub,
        overlaps,
    )
    crops = get_crops_recurse(bounds)
    entries = []
    always_predict = False
    for idx, crop in enumerate(crops):
        print('prediction partial {:03d}/{:03d} | crop: {}'.format(idx + 1, len(crops), crop))
        slices = [slice(None), slice(None)] + [slice(start, end) for start, end in crop]
        batch_x_partial = batch_x[slices]
        path_save_partial = os.path.join(path_save_partials_dir, 'partial_{:03d}.tif'.format(idx))
        if always_predict or not os.path.exists(path_save_partial):
            batch_prediction_partial = model.predict(batch_x_partial)
            tifffile.imsave(path_save_partial, batch_prediction_partial[0, 0,])
            print('saved tif:', path_save_partial)
        else:
            print('{:s} exists. skipping....'.format(path_save_partial))
        entry = {
            'path_partial': path_save_partial,
            'start_z': crop[0][0],
            'end_z': crop[0][1],
            'start_y': crop[1][0],
            'end_y': crop[1][1],
            'start_x': crop[2][0],
            'end_x': crop[2][1],
        }
        entries.append(entry)
    path_partials_csv = os.path.join(path_save_partials_dir, 'partials.csv')
    pd.DataFrame(entries).to_csv(path_partials_csv, index=False)
    print('saved csv:', path_partials_csv)
    batch_prediction = np.zeros(batch_x.shape, batch_x.dtype)
    ar_canvas = batch_prediction[0, 0, ]
    combine_partials(path_save_partials_dir, ar_canvas)
    return batch_prediction

def old_combine_partials(path_partials_dir, ar_canvas):
    # read partials.csv for the locations corresponding to each partial; place partial in cavnas
    print('combining partials in:', path_partials_dir)
    df = pd.read_csv(os.path.join(path_partials_dir, 'partials.csv'))
    for idx, row in df.iterrows():
        slices = [
            slice(row['start_z'], row['end_z']),
            slice(row['start_y'], row['end_y']),
            slice(row['start_x'], row['end_x']),
        ]
        path_partial = row['path_partial']
        ar_partial = tifffile.imread(path_partial)
        ar_canvas[slices] = ar_partial
        print('added:', path_partial)

def combine_partials(path_partials_dir, ar_canvas):
    # read partials.csv for the locations corresponding to each partial; place partial in cavnas
    def find_start_offset(row, dim, ends_unq, dict_results):
        # for a given dimension, find some other end bound that falls between the start-end bounds of that dimension
        # dim is one of ['z', 'y', 'z']
        lower = row['start_' + dim]
        upper = row['end_' + dim]
        key = (dim, lower, upper)
        if key in dict_results:
            return dict_results[key]
        ends_others = ends_unq[dim]
        offset = 0
        for end_other in ends_others:
            if lower < end_other < upper:
                offset = (end_other - lower)//2
                break
        dict_results[key] = offset
        return offset
    
    print('combining partials in:', path_partials_dir)
    df = pd.read_csv(os.path.join(path_partials_dir, 'partials.csv'))
    ends_unq = {}
    for dim in 'xyz':
        ends_unq[dim] = df['end_' + dim].unique()
    # look for end_* that fall between start_* and end_*
    dict_results = {}
    update_df = {
        'offset_start_z': [],
        'offset_start_y': [],
        'offset_start_x': [],
    }
    for idx, row in df.iterrows():
        slices_canvas = []
        slices_partial = []
        entry = {}
        for dim in 'zyx':
            offset_start = find_start_offset(row, dim, ends_unq, dict_results)
            start = offset_start + row['start_' + dim]
            slices_canvas.append(
                slice(start, row['end_' + dim]),
            )
            slices_partial.append(
                slice(offset_start, None)
            )
            update_df['offset_start_' + dim].append(offset_start)
        path_partial = row['path_partial']
        ar_partial = tifffile.imread(path_partial)
        ar_canvas[slices_canvas] = ar_partial[slices_partial]
        print('added:', path_partial)
    path_save_update_csv = os.path.join(path_partials_dir, 'partials_combination_record.csv')
    df.assign(**update_df).to_csv(path_save_update_csv, index=False)
    print('wrote csv:', path_save_update_csv)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--n_images', type=int, default=16, help='max number of images to test')
    parser.add_argument('--path_save_dir', default='results', help='path to output directory')
    parser.add_argument('--path_source', help='path to CSV of model/dataset paths or path to run directory')
    parser.add_argument('--overlap', type=int, default=16, help='overlap between prediction partials')
    parser.add_argument('--pixels_max', type=int, default=9732096, help='max number of pixels in input volumes')
    parser.add_argument('--img_sel', type=int, nargs='+', help='select elements of dataset')
    opts = parser.parse_args()
    
    df_models = get_df_models(opts.path_source)
    for idx_model, model_info in df_models.iterrows():
        dataset = fnet.data.load_dataset_from_json(model_info['path_dataset'])
        print('*** DataSet ***')
        print(dataset)
        dataprovider = fnet.data.TestImgDataProvider(dataset, transforms=None)
        dataprovider.use_test_set()
        model = None
        if model_info['path_model'] is not None:
            model = model_module.Model(
                gpu_ids=opts.gpu_ids,
            )
            model.load_state(model_info['path_model'])
        print('*** Model ***')
        print(model)
        
        tag = 'none' if model is None else '{:05d}'.format(model.count_iter)
        name_model = model_info.get('name_model', 'no_name')
        path_save_dir_this_model = os.path.join(opts.path_save_dir, 'output_{:s}_{:s}'.format(
            name_model,
            tag,
        ))
        indices = range(len(dataprovider)) if opts.img_sel is None else opts.img_sel
        for idx in indices:
            print('DEBUG: doing item', idx)
            batch_x, batch_y = dataprovider[idx]
            pdb.set_trace()
            path_save_partial = os.path.join(
                path_save_dir_this_model,
                'img_{:s}_{:02d}'.format('train' if dataprovider.using_train_set() else 'test', idx)
            )
            kwargs = {
                'pixels_max': opts.pixels_max,
                'path_save_partials_dir': path_save_partial + '_prediction_partials',
                'overlaps': [opts.overlap]*3,
            }
            batch_prediction = get_batch_prediction(model, batch_x, **kwargs)
            save_results(
                ar_signal = batch_x[0, 0,],
                ar_target = batch_y[0, 0,] if batch_y is not None else None,
                ar_prediction = batch_prediction[0, 0,],
                path_save_partial = path_save_partial,
            )

            entry = {
                'path_czi': dataprovider.get_name(idx),
                'index': idx,
                'test_or_train': 'train' if dataprovider.using_train_set() else 'test',
                'l2': 999,
                'l2_norm': 888,
            }
            print('DEBUG: entry', entry)
            # if False:
            #     pd.concat([df_losses_per_test, df_losses_per_train]).to_csv(
            #         os.path.join(path_out, 'results_{:s}_per.csv'.format(name_model)),
            #         index=False,
            #     )
                

def save_results(
            ar_signal = None,
            ar_target = None,
            ar_prediction = None,
            path_save_partial = None,
        ):
    dirname = os.path.dirname(path_save_partial)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if ar_target is None:
        print('making dummy target:')
        ar_target = np.zeros(ar_signal.shape, dtype=ar_signal.dtype)
    path_save_signal = path_save_partial + '_signal.tif'
    path_save_target = path_save_partial + '_target.tif'
    path_save_prediction = path_save_partial + '_prediction.tif'
    compression = 0
    tifffile.imsave(path_save_signal, ar_signal.astype(np.float32), compress=compression)
    print('saved tif:', path_save_signal)
    tifffile.imsave(path_save_target, ar_target.astype(np.float32), compress=compression)
    print('saved tif:', path_save_target)
    tifffile.imsave(path_save_prediction, ar_prediction.astype(np.float32), compress=compression)
    print('saved tif:', path_save_prediction)

def test():
    dim_source = (49, 3104, 656)
    # dim_source = (48, 3104, 448)
    shape_sub = (48, 448, 448)
    # overlaps = (16, 16, 16)
    overlaps = (32, )*3
    bounds = calc_bounds(
        dim_source,
        shape_sub,
        overlaps,
    )
    print(bounds)
    print('*** crops ***')
    tmp = get_crops_recurse(bounds[-3:])
    for i, t in enumerate(tmp):
        print(i, t)
    
def test_stitch():
    ar_canvas = np.zeros((49, 3104, 656))
    path_partials_dir = 'results/rnd/etf/large_01/output_trial_00_50000/img_test_00_prediction_partials'
    rnd_combine_partials(path_partials_dir, ar_canvas)
    
    
if __name__ == '__main__':
    main()
    # test()
    # test_stitch()
