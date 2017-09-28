import argparse
import importlib
import fnet.data
import fnet.display
import numpy as np
import os
import torch
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', help='path to data directory')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--load_path', help='path to trained model')
parser.add_argument('--model_module', default='ttf_model', help='name of the model module')
parser.add_argument('--n_images', type=int, help='max number of images to test')
parser.add_argument('--build_z_animation', action='store_true', default=False, help='save each z slice of test images')
parser.add_argument('--save_images', action='store_true', default=False, help='save test image results')
parser.add_argument('--use_train_set', action='store_true', default=False, help='view predictions on training set images')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='enable verbose output')
opts = parser.parse_args()

model_module = importlib.import_module('model_modules.'  + opts.model_module)

# from past run through data:
# mean of means bright-field: 0.0282992
# mean of means prediction:   421.485
# mean of stds bright-field:  0.981663
# mean of stds prediction:    13.1872
def test_display(model, data):
    y_pred = np.zeros((1, 1) + data.get_dims_chunk(), dtype=np.float32)
    means_bf = []
    means_pred = []
    stds_bf = []
    stds_pred = []

    mean_means_bf = 0.0282992
    mean_means_pred = 421.485
    mean_stds_bf = 0.981663
    mean_stds_pred = 13.1872
    # z_selectors = [10, 12, 14, 18, 20, 22]
    z_selectors = [2, 16]
    for i, (x_test, y_true) in enumerate(data):
        if model is not None:
            y_pred[:] = model.predict(x_test)
        path_z_ani = None
        if opts.build_z_animation:
            path_z_ani = 'presentation/' + ('test' if not opts.use_train_set else 'train') + '_{:02d}'.format(i)
        sources = (x_test, y_pred)
        titles = ('bright-field', 'prediction')
        vmins = (mean_means_bf -  8*mean_stds_bf, mean_means_pred -  1.7*mean_stds_pred)
        vmaxs = (mean_means_bf +  8*mean_stds_bf, mean_means_pred + 9*mean_stds_pred)
        
        for z_selector in z_selectors:
            path_save_pre = 'presentation/z{:02d}/timelapse_{:02d}'.format(z_selector, i)
            path_dirname = os.path.dirname(path_save_pre)
            if not os.path.exists(path_dirname):
                os.makedirs(path_dirname)
            fnet.display.display_visual_eval_images(sources,
                                                    z_selector=z_selector,
                                                    titles=titles,
                                                    vmins=vmins,
                                                    vmaxs=vmaxs,
                                                    verbose=opts.verbose,
                                                    path_save_pre=path_save_pre,
                                                    path_z_ani=path_z_ani)
        if opts.save_images:
            name_model = os.path.basename(opts.load_path).split('.')[0]
            img_trans = x_test[0, 0, ].astype(np.float32)
            img_dna = y_true[0, 0, ].astype(np.float32)
            img_pred = y_pred[0, 0, ]
            name_pre = 'test_output/{:s}_test_{:02d}_'.format(name_model, i)
            name_post = '.tif'
            name_trans = name_pre + 'trans' + name_post
            name_dna = name_pre + 'dna' + name_post
            name_pred = name_pre + 'prediction' + name_post
            fnet.save_img_np(img_trans, name_trans)
            fnet.save_img_np(img_dna, name_dna)
            fnet.save_img_np(img_pred, name_pred)

        # means_bf.append(np.mean(x_test[0, 0, z_selector, ]))
        # means_pred.append(np.mean(y_pred[0, 0, z_selector, ]))
        # stds_bf.append(np.std(x_test[0, 0, z_selector, ]))
        # stds_pred.append(np.std(y_pred[0, 0, z_selector, ]))
        
        if (opts.n_images is not None) and (i == (opts.n_images - 1)):
            break
    # print('mean of means bright-field:', np.mean(means_bf))
    # print('mean of means prediction:', np.mean(means_pred))
    # print('mean of stds bright-field:', np.mean(stds_bf))
    # print('mean of stds prediction:', np.mean(stds_pred))

    
def main():
    torch.cuda.set_device(opts.gpu_id)
    print('on GPU:', torch.cuda.current_device())
    
    if opts.use_train_set:
        print('*** Using training set ***')
    train_select = opts.use_train_set

    # load test dataset
    dataset = fnet.data.DataSet(opts.data_path, train_select=train_select)
    print(dataset)

    dims_chunk = (32, 208, 208)
    # dims_chunk = (48, 224, 320)
    dims_pin = (0, 0, 0)
    data_test = fnet.data.TestImgDataProvider(dataset, dims_chunk=dims_chunk, dims_pin=dims_pin)
    
    # load model
    model = None
    if opts.load_path is not None:
        model = model_module.Model(load_path=opts.load_path)
    print(model)
    test_display(model, data_test)

if __name__ == '__main__':
    main()
