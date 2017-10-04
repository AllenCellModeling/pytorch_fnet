import argparse
import importlib
import fnet
import fnet.data
import fnet.data.transforms
import fnet.data
import pandas as pd
import os
import warnings
import pdb

def get_df_models(opts):
    if opts.path_source.lower().endswith('.csv'):
        df_models = pd.read_csv(opts.path_source)
    else:
        assert os.path.isdir(opts.path_source)
        model_info = {}
        for entry in os.scandir(opts.path_source):
            if entry.path.lower().endswith('model.p'):
                model_info['path_model'] = [entry.path]
            elif entry.path.lower().endswith('ds.json'):
                model_info['path_dataset'] = [entry.path]
        df_models = pd.DataFrame(model_info)
    return df_models

def main():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--model_module', default='fnet_model', help='name of the model module')
    parser.add_argument('--n_images', type=int, default=16, help='max number of images to test')
    parser.add_argument('--path_source', help='path to CSV of model/dataset paths or path to run directory')
    parser.add_argument('--path_save_dir', default='results', help='path to output directory')
    opts = parser.parse_args()
    
    model_module = importlib.import_module('model_modules.'  + opts.model_module)

    df_models = get_df_models(opts)

    results_list = []
    for idx, model_info in df_models.iterrows():
        # load test dataset
        dataset = fnet.data.load_dataset_from_json(model_info['path_dataset'])
        print(dataset)
        dims_cropped = (32, '/16', '/16')
        cropper = fnet.data.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
        transforms = (cropper, cropper)
        dataprovider = fnet.data.TestImgDataProvider(dataset, transforms)

        # load model
        model = None
        if model_info['path_model'] is not None:
            model = model_module.Model(
                gpu_ids=opts.gpu_ids,
            )
            model.load_state(model_info['path_model'])
        print('model:', model)

        tag = 'none' if model is None else '{:05d}'.format(model.count_iter)
        path_out = os.path.join(opts.path_save_dir, 'output_{:s}'.format(tag))
        dataprovider.use_test_set()
        losses_test = pd.Series(fnet.test_model(
            model,
            dataprovider,
            n_images = opts.n_images,
            path_save_dir = path_out,
        ))
        dataprovider.use_train_set()
        losses_train = pd.Series(fnet.test_model(
            model,
            dataprovider,
            n_images = opts.n_images,
            path_save_dir = path_out,
        ))
        
        results_entry = pd.concat([model_info, losses_test, losses_train])
        results_list.append(results_entry)
        df_results = pd.DataFrame(results_list)
        path_out_csv = os.path.join(path_out, 'results.csv')
        df_results.to_csv(path_out_csv, index=False)
        
    
if __name__ == '__main__':
    main()
