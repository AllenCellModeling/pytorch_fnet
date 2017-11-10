import argparse
import tifffile
import numpy as np
import fnet.data
import fnet.data.transforms
import model_modules.fnet_model
import os
import pdb
import json
import pandas as pd
import time

MAP_TAGS_TO_MODELS = {
    'alpha_tubulin': 'saved_models/alpha_tubulin',
    'beta_actin': 'saved_models/beta_actin',
    'desmoplakin': 'saved_models/desmoplakin',
    'dic_lamin_b1': 'saved_models/dic_lamin_b1',
    'dic_membrane': 'saved_models/dic_membrane',
    'dna': 'saved_models/dna',
    'fibrillarin': 'saved_models/fibrillarin',
    'lamin_b1': 'saved_models/lamin_b1',
    'membrane': 'saved_models/membrane',
    'myosin_iib': 'saved_models/myosin_iib',
    'sec61_beta': 'saved_models/sec61_beta',
    'tom20': 'saved_models/tom20',
    'zo1': 'saved_models/zo1',
}

class GhettoIntegratedCells(object):
    def __init__(self, paths_models, gpu_id=0):
        self.paths_models = paths_models
        self.gpu_id = gpu_id
        self.process_models()  # self.paths_load, self.names, self.df_training_czis
        print('***** models *****')
        for path in paths_models:
            print(path)

    def path_in_training_set_of(self, path_czi):
        """Return list of model names that used path_czi in their training set."""
        mask = self.df_training_czis['path_czi'].str.contains(path_czi)
        return list(self.df_training_czis[mask]['name'].unique())

    def get_name(self, idx_model):
        return self.names[idx_model]

    def process_models(self):
        def is_train_set_csv(path_csv):
            if path_csv.lower().endswith('.csv'):
                if 'train' in path_csv.lower():
                    return True
            return False
        
        paths_load = []
        names = []
        df_training_czis = pd.DataFrame()
        paths_train_sets = []
        for path_model in paths_models:
            for path_file in [i.path for i in os.scandir(path_model)]:
                if path_file.lower().endswith('model.p'):
                    name = os.path.basename(path_model)
                    path_load = path_file
                if is_train_set_csv(path_file):
                    df_add = pd.read_csv(path_file)
                    df_add['name'] = name
                    df_training_czis = pd.concat([df_training_czis, df_add], ignore_index=True)
                    paths_train_sets.append(path_file)
            names.append(name)
            paths_load.append(path_load)
        self.names = names
        self.paths_load = paths_load
        self.paths_train_sets = paths_train_sets
        self.df_training_czis = df_training_czis

    def get_prediction(self, x_signal, idx_model):
        assert isinstance(x_signal, np.ndarray)
        assert x_signal.ndim == 5
        predictions = []
        path_model = self.paths_models[idx_model]
        if path_model is None:
            prediction = np.zeros((x_signal.shape), dtype=x_signal.dtype)
        else:
            path_load = self.paths_load[idx_model]
            model = model_modules.fnet_model.Model(gpu_ids=self.gpu_id)
            model.load_state(path_load)
            print('predicting', self.names[idx_model])
            prediction = model.predict(x_signal)
        return prediction
    
def get_sources_from_files(path):
    def order(fname):
        if 'bf' in fname: return 0
        if 'lamin' in fname: return 1
        if 'fibrillarin' in fname: return 2
        if 'tom' in fname: return 3
        if 'all' in fname: return 10
        return 0
    files = [i.path for i in os.scandir(path) if i.is_file()]
    paths_sources = [i for i in files if (('rgb.tif' in i) or ('bf.tif' in i))]
    paths_sources.sort(key=order)
    sources = []
    for path in paths_sources:
        source = tifffile.imread(path)
        sources.append(source)
    return sources
    
def get_dataset_from_source(path_source):
    if path_source.lower().endswith('.json'):
        ds = fnet.data.load_dataset_from_json(path_source)
    elif path_source.lower().endswith('.csv'):
        transform = fnet.data.transforms.sub_mean_norm
        ds = fnet.data.DataSet(
            None,
            path_source,
            transforms = [transform, transform],
        )
    elif os.path.isdir(path_source):
        raise NotImplementedError
    elif path_source.lower().endswith('.czi'):
        pass
    else:
        raise NotImplementedError
    ds.use_test_set()
    return ds
            
    if path_source.lower().endswith('.czi'):
        # aiming for 0.3 um/px
        z_fac = 0.97
        xy_fac = 0.217  # timelapse czis; original scale 0.065 um/px
        resize_factors = (z_fac, xy_fac, xy_fac)
        resizer = fnet.data.transforms.Resizer(resize_factors)
        transforms = ((resizer, fnet.data.transforms.sub_mean_norm),
                      (resizer, fnet.data.transforms.sub_mean_norm))
        dataset = fnet.data.functions.create_dataset(
            path_source=path_source,
            name_signal='bf',
            name_target='struct',
            train_split=0,
            transforms=transforms,
        )
    else:
        dataset = fnet.data.functions.load_dataset(path_source)
    dataset.use_test_set()
    return dataset

def predict_multi_model(paths_models, dataset, indices, n_images, path_save_dir, gpu_id):
    time_start = time.time()
    dims_cropped = ('/16', '/16', '/16')
    cropper = fnet.data.transforms.Cropper(dims_cropped, offsets=('mid', 'mid', 'mid'))
    transforms = (cropper, cropper)
    data_test = fnet.data.TestImgDataProvider(dataset, transforms)
    
    gic = GhettoIntegratedCells(paths_models, gpu_id)
    
    # color/title setup
    n_models = len(paths_models)
    count = 0
    list_predicted = []
    for ind in indices:
        print('dataset element {:d}/{:d} | image {:d}'.format(ind, len(data_test), count))
        path_czi = data_test.get_name(ind)
        in_trainig_set_of = gic.path_in_training_set_of(path_czi)
        if len(in_trainig_set_of) > 0:
            print('path_czi:', path_czi)
            print('element is in training set of {:s}. skipping....'.format(str(in_trainig_set_of)))
            continue
        signal, target = data_test[ind]
        for idx_save in range(n_models + 2):
            if idx_save == 0:
                name = 'signal'
                ar_save = signal[0, 0, ].astype(np.float32)
            elif idx_save == 1:
                name = 'target'
                ar_save = target[0, 0, ].astype(np.float32)
            else:
                idx_model = idx_save - 2
                name = gic.get_name(idx_model)
                prediction = gic.get_prediction(signal, idx_model)
                ar_save = prediction[0, 0, ].astype(np.float32)
            path_element_dir = os.path.join(path_save_dir, '{:04d}'.format(count))
            if not os.path.exists(path_element_dir):
                os.makedirs(path_element_dir)
            path_save = os.path.join(path_element_dir, '{:04d}_{:s}_gray.tif'.format(count, name))
            tifffile.imsave(path_save, ar_save, photometric='minisblack')
            print('saved tif to:', path_save)
            print('elapsed time: {:.2f}'.format(time.time() - time_start))
        list_predicted.append({'id':count, 'path_czi':path_czi})
        count += 1
        if (n_images is not None) and (count >= n_images):
            break
    path_csv = os.path.join(path_save_dir, 'source_czis.csv')
    pd.DataFrame(list_predicted).to_csv(path_csv, index=False)
    print('wrote csv:', path_csv)

def make_gif(path_source,
             timelapse=False,
             delay=20,
             z_slice=13,
):
    """
    path_source - directory of source file/directory
    timelapse - (bool) 
    z_slice - (int) only used for timelapse gifs
    """
    if timelapse:
        assert os.path.isdir(path_source)
        path_source_base = os.path.basename(path_source)
        source_str = '"{:s}*_all_rgb.tif[{:d}]"'.format(
            path_source + os.path.sep,
            z_slice
        )
        dst_str = '{:s}gic_{:s}_z{:02d}.gif'.format(
            path_source + os.path.sep,
            path_source_base,
            z_slice
        )
        cmd_str = 'convert -delay {:d} {:s} {:s}'.format(
            delay,
            source_str,
            dst_str
        )
        print(cmd_str)
    else:
        raise NotImplementedError
    subprocess.run(cmd_str, shell=True, check=True)

def get_settings_from_json(path_json):
    with open(path_json, 'r') as fi:
        settings = json.load(fi)
    return settings

if __name__ == '__main__':
    """
    path_source - CZI, dataset, or CSV
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_source', help='path to data CZI or saved dataset')
    parser.add_argument('--path_save_dir', help='path to save directory')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--img_sel', type=int, nargs='*', help='select images to test')
    parser.add_argument('--shuffle', action='store_true', help='set to shuffle input images')
    parser.add_argument('--n_images', type=int, help='maximum number of images to process')
    parser.add_argument('--path_json', help='path to settings json ')
    parser.add_argument('--tags_models', nargs='+', default=['dna', 'fibrillarin', 'lamin_b1', 'sec61_beta', 'tom20'], help='maximum number of images to process')
    opts = parser.parse_args()

    map_tags_to_models = MAP_TAGS_TO_MODELS
    if opts.path_json is not None:
        settings = get_settings_from_json(opts.path_json)
        if 'map_tags_to_models' in settings:
            map_tags_to_models = settings['map_tags_to_models']
    
    paths_models = [map_tags_to_models[i] for i in opts.tags_models]
    dataset = get_dataset_from_source(opts.path_source)
    print(dataset)

    if not os.path.exists(opts.path_save_dir):
        os.makedirs(opts.path_save_dir)
    path_save_options = os.path.join(opts.path_save_dir, 'options.json')
    with open(path_save_options, 'w') as fo:
        json.dump(vars(opts), fo, indent=4)
        print('wrote:', path_save_options)
    indices = list(range(len(dataset))) if opts.img_sel is None else opts.img_sel
    if opts.shuffle:
        np.random.shuffle(indices)
    predict_multi_model(paths_models, dataset, indices, opts.n_images, opts.path_save_dir, opts.gpu_id)
