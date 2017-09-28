import argparse
import importlib
import fnet
import fnet.data
import fnet.display
import numpy as np
import os
import torch
import warnings
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', help='path to data directory')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--load_path', help='path to trained model')
parser.add_argument('--model_module', default='snm_model', help='name of the model module')
parser.add_argument('--n_images', type=int, help='max number of images to test')
parser.add_argument('--save_each_slice', action='store_true', default=False, help='save each z slice of test image')
parser.add_argument('--save_images', action='store_true', default=False, help='save test image results')
parser.add_argument('--use_train_set', action='store_true', default=False, help='view predictions on training set images')
opts = parser.parse_args()

warnings.filterwarnings('ignore', message='.*zoom().*')  # TODO: replace with write to file
warnings.filterwarnings('ignore', message='.*end of stream*')  # TODO: replace with write to file
model_module = importlib.import_module('model_modules.'  + opts.model_module)

def test_display(model, data):
    y_pred = np.zeros((1, 1) + data.get_dims_chunk(), dtype=np.float32)
    titles = ('DNA', 'combi-mask', 'predicted')
    for i, (x_test, y_true) in enumerate(data):
        if model is not None:
            y_pred[:] = model.predict(x_test)
        sources = (x_test, y_true, y_pred)
        z_max = fnet.find_z_of_max_slice(x_test[0, 0, ])
        fnet.display.display_visual_eval_images(sources,
                                                z_display=z_max,
                                                titles=titles,
                                                vmins=None,
                                                vmaxs=None,
                                                verbose=False,
                                                path_z_ani=None)
        if opts.save_images:
            path_base = os.path.join('test_output', os.path.basename(opts.load_path).split('.')[0])
            for idx, source in enumerate(sources):
                img = source[0, 0, ].astype(np.float32)
                path_img = os.path.join(path_base, 'img_{:02d}_{:s}.tif'.format(i, titles[idx]))
                print('saving to:', path_img)
                fnet.save_img_np(img, path_img)
        if opts.save_each_slice:
            dir_save = 'presentation/' + ('test' if not opts.use_train_set else 'train') + '_{:02d}'.format(i)
            fnet.display.save_image_stacks(dir_save, (x_test, y_true, y_pred))
        if (opts.n_images is not None) and (i == (opts.n_images - 1)):
            break
    
    
def main():
    torch.cuda.set_device(opts.gpu_id)
    print('on GPU:', torch.cuda.current_device())
    
    if opts.use_train_set:
        print('*** Using training set ***')
    train_select = opts.use_train_set

    # load test dataset
    dataset = fnet.data.NucMaskDataSet(opts.data_path, train_select=train_select)
    print(dataset)

    # dims_chunk = (32, 224, 224)
    dims_chunk = (32, 208, 208)
    dims_pin = (0, 0, 0)
    data_test = fnet.data.TestImgDataProvider(dataset, dims_chunk=dims_chunk, dims_pin=dims_pin)
    
    # load model
    model = None
    if opts.load_path is not None:
        model = model_module.Model(load_path=opts.load_path)
    print('model:')
    print(model)
    test_display(model, data_test)

if __name__ == '__main__':
    main()
