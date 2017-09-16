import argparse
import importlib
import util
import util.data
import util.data.transforms
import util.data
import pandas as pd
import os
import warnings
import pdb

def get_df_models(opts):
    if opts.path_source is None:
        assert opts.path_dataset is not None and opts.path_model is not None
        paths_models = [opts.path_model]
        paths_datasets = [opts.path_dataset]
        models = []
        model_info = {
            'path_model': opts.path_model,
            'path_dataset': opts.path_dataset,
        }
        models.append(model_info)
        # dummy = {
        #     'path_model': None,
        #     'path_dataset': opts.path_dataset,
        # }
        # models.append(dummy)
        df_models = pd.DataFrame(models)
    else:
        if opts.path_source.lower().endswith('.csv'):
            df_models = pd.read_csv(opts.path_source)
        else:
            assert os.path.isdir(opts.path_source)
            model_info = {}
            for entry in os.scandir(opts.path_source):
                if entry.path.lower().endswith('.p'):
                    model_info['path_model'] = [entry.path]
                elif entry.path.lower().endswith('.ds'):
                    model_info['path_dataset'] = [entry.path]
            df_models = pd.DataFrame(model_info)
    return df_models

def main():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--img_sel', type=int, nargs='*', help='select images to test')
    parser.add_argument('--model_module', default='ttf_model', help='name of the model module')
    parser.add_argument('--n_images', type=int, help='max number of images to test')
    parser.add_argument('--path_csv_out', default='test_output/model_losses.csv', help='path to output CSV')
    parser.add_argument('--path_dataset', help='path to data directory')
    parser.add_argument('--path_model', help='path to trained model')
    parser.add_argument('--path_save', help='path to directory where output should be saved')
    parser.add_argument('--path_source', help='path to CSV of model/dataset paths or path to run directory')
    parser.add_argument('--save_images', action='store_true', default=False, help='save test image results')
    parser.add_argument('--use_train_set', action='store_true', default=False, help='view predictions on training set images')
    opts = parser.parse_args()
    
    model_module = importlib.import_module('model_modules.'  + opts.model_module)

    df_models = get_df_models(opts)

    results_list = []
    for idx, model_info in df_models.iterrows():
        print(model_info, type(model_info))

        path_model = model_info['path_model']
        path_dataset = model_info['path_dataset']
    
        # load test dataset
        dataset = util.data.functions.load_dataset(path_dataset)
        print(dataset)

        dims_cropped = (32, '/16', '/16')
        cropper = util.data.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
        transforms = (cropper, cropper)
        dataprovider = util.data.TestImgDataProvider(dataset, transforms)

        # load model
        model = None
        if path_model is not None:
            model = model_module.Model(load_path=path_model,
                                       gpu_ids=opts.gpu_ids
            )
        print('model:', model)
        dataprovider.use_test_set()
        losses_test = pd.Series(util.test_model(model, dataprovider, opts))
        dataprovider.use_train_set()
        losses_train = pd.Series(util.test_model(model, dataprovider, opts))
        
        results_entry = pd.concat([model_info, losses_test, losses_train])
        results_list.append(results_entry)
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(opts.path_csv_out, index=False)
    print(df_results.loc[:, ['path_model', 'path_dataset', 'l2_norm_train', 'l2_norm_test']])
    
if __name__ == '__main__':
    main()
