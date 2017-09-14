import argparse
import importlib
import util.data
import util.data.transforms
import util.data.functions
import util.display
import numpy as np
import pandas as pd
import os
import torch.autograd
import torch.nn
import warnings
import subprocess
import pdb

warnings.filterwarnings('ignore', message='.*zoom().*')
warnings.filterwarnings('ignore', message='.*end of stream*')
warnings.filterwarnings('ignore', message='.*multiple of element size.*')

def get_losses_norm(prediction, target):
    l1_norm = np.sum(np.absolute(target))
    l2_norm = np.sum(np.square(target))
    l1_loss_norm = np.sum(np.absolute(prediction - target))/l1_norm
    l2_loss_norm = np.sum(np.square(prediction - target))/l2_norm
    return l1_loss_norm, l2_loss_norm

def get_losses(prediction, target):
    y_pred = torch.autograd.Variable(torch.Tensor(prediction), volatile=True)
    y_targ = torch.autograd.Variable(torch.Tensor(target), volatile=True)
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    l1_loss_float = float(l1_loss(y_pred, y_targ).data.numpy())
    l2_loss_float = float(l2_loss(y_pred, y_targ).data.numpy())
    return l1_loss_float, l2_loss_float

def test_model(model, data, opts):
    indices = range(len(data)) if opts.img_sel is None else opts.img_sel
    l1_losses, l2_losses = [], []
    l1_norm_losses, l2_norm_losses = [], []
    test_or_train = 'train' if data.using_train_set() else 'test'
    titles = ('signal', 'target', 'prediction')
    if opts.path_save is not None:
        path_base = opts.path_save
    elif os.path.isdir(opts.path_source):
        path_base = os.path.join(opts.path_source, 'outputs')
    elif opts.path_model is not None:
        path_base = os.path.join('test_output', os.path.basename(opts.path_model).split('.')[0])
    else:
        path_base = os.path.join('test_output', os.path.basename(opts.path_dataset).split('.')[0])

    for i in indices:
        vmins, vmaxs = None, None
        try:
            x_test, y_true = data[i]
        except AttributeError:
            print('skipping....')
            continue
        if model is not None:
            y_pred = model.predict(x_test)
        else:
            y_pred = np.zeros(x_test.shape, dtype=np.float64)
        sources = (x_test[0, 0, ], y_true[0, 0, ], y_pred[0, 0, ])

        l1_loss, l2_loss = get_losses(y_pred, y_true)
        l1_norm_loss, l2_norm_loss = get_losses_norm(y_pred, y_true)
        l1_losses.append(l1_loss)
        l2_losses.append(l2_loss)
        l1_norm_losses.append(l1_norm_loss)
        l2_norm_losses.append(l2_norm_loss)
        print('{:s} image: {:02d} | name: {} | l1: {:.4f} | l2: {:.4f} | l1_norm: {:.4f} | l2_norm: {:.4f}'.format(
            test_or_train,
            i,
            data.get_name(i),
            l1_loss,
            l2_loss,
            l1_norm_loss,
            l2_norm_loss,
        ))
        path_img_pre = os.path.join(path_base, 'img_{:s}_{:02d}'.format(test_or_train, i))
        if opts.save_images:
            for idx, source in enumerate(sources):
                img = source.astype(np.float32)
                path_img = path_img_pre + '_{:s}.tif'.format(titles[idx])
                util.save_img_np(img, path_img)
        if (opts.n_images is not None) and (i >= (opts.n_images - 1)):
            break
    l1_loss_mean = np.mean(l1_losses)
    l2_loss_mean = np.mean(l2_losses)
    l1_norm_loss_mean = np.mean(l1_norm_losses)
    l2_norm_loss_mean = np.mean(l2_norm_losses)
    print('*****')
    print('count: {:d} | l1: {:.4f} | l2: {:.4f} | l1_norm: {:.4f} | l2_norm: {:.4f}'.format(
        len(l1_losses),
        l1_loss_mean,
        l2_loss_mean,
        l1_norm_loss_mean,
        l2_norm_loss_mean
    )
    )
    l2_norm_argsort = np.argsort(l2_norm_losses)
    print('Sorted by L2 norm loss')
    for i in l2_norm_argsort:
        print('{:s} {:02d} | L2 loss: {:.4f}'.format(test_or_train, i, l2_norm_losses[i]))
    ret_dict = {
        'l1_' + test_or_train: l1_loss_mean,
        'l2_' + test_or_train: l2_loss_mean,
        'l1_norm_' + test_or_train: l1_norm_loss_mean,
        'l2_norm_' + test_or_train: l2_norm_loss_mean,
        'n_count_' + test_or_train: len(l1_losses),
    }
    return ret_dict

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
        losses_test = pd.Series(test_model(model, dataprovider, opts))
        dataprovider.use_train_set()
        losses_train = pd.Series(test_model(model, dataprovider, opts))
        
        results_entry = pd.concat([model_info, losses_test, losses_train])
        results_list.append(results_entry)
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(opts.path_csv_out, index=False)
    print(df_results.loc[:, ['path_model', 'path_dataset', 'l2_norm_train', 'l2_norm_test']])
    
if __name__ == '__main__':
    main()
