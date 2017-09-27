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
        assert opts.path_model is not None and opts.path_data_train is not None and opts.path_data_test is not None 
        models = []
        model_info = {
            'path_model': opts.path_model,
            'path_data_train': opts.path_data_train,
            'path_data_test': opts.path_data_test,
        }
        models.append(model_info)
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
                elif entry.path.lower().endswith('.csv'):
                    if 'train' in os.path.basename(entry.path):
                        model_info['path_data_train'] = [entry.path]
                    if 'test' in os.path.basename(entry.path):
                        model_info['path_data_test'] = [entry.path]
            df_models = pd.DataFrame(model_info)
    return df_models

def main():
    warnings.filterwarnings('ignore', message='.*zoom().*')
    warnings.filterwarnings('ignore', message='.*end of stream*')
    warnings.filterwarnings('ignore', message='.*multiple of element size.*')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
    parser.add_argument('--img_sel', type=int, nargs='*', help='select images to test')
    parser.add_argument('--model_module', default='fnet_model', help='name of the model module')
    parser.add_argument('--n_images', type=int, help='max number of images to test')
    parser.add_argument('--path_csv_out', default='test_output/model_losses.csv', help='path to output CSV')
    parser.add_argument('--path_data_test', help='path to test set csv')
    parser.add_argument('--path_data_train', help='path to training set csv')
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
        # load test dataset
        dataset = util.data.functions.load_dataset(
            path_data_train = model_info['path_data_train'],
            path_data_test = model_info['path_data_test'],
        )
        print(dataset)
        dims_cropped = (32, '/16', '/16')
        cropper = util.data.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
        transforms = (cropper, cropper)
        dataprovider = util.data.TestImgDataProvider(dataset, transforms)

        # load model
        model = None
        if model_info['path_model'] is not None:
            model = model_module.Model(gpu_ids=opts.gpu_ids)
            model.load_state(model_info['path_model'])
        print('model:', model)
        dataprovider.use_test_set()

        losses_test = pd.Series(util.test_model(model, dataprovider, **vars(opts)))
        dataprovider.use_train_set()
        losses_train = pd.Series(util.test_model(model, dataprovider, **vars(opts)))
        
        results_entry = pd.concat([model_info, losses_test, losses_train])
        results_list.append(results_entry)
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(opts.path_csv_out, index=False)
    print(df_results.loc[:, ['path_model', 'path_data_train', 'path_data_test', 'l2_norm_train', 'l2_norm_test']])
    
if __name__ == '__main__':
    main()
