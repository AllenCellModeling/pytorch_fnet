import argparse
import importlib
import util.data
import util.display
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

def test_display(model, data):
    for i, (x_test, y_true) in enumerate(data):
        if model is not None:
            y_pred = model.predict(x_test)
        else:
            y_pred = np.zeros(x_test.shape, dtype=np.float32)
        path_z_ani = None
        if opts.build_z_animation:
            path_z_ani = 'presentation/' + ('test' if not opts.use_train_set else 'train') + '_{:02d}'.format(i)
        # z_selector = 'strongest_in_target'  # select z based on DNA channel
        sources = (x_test, y_true, y_pred)
        z_display = 'strongest_1'
        titles = ('bright-field', 'DNA', 'prediction')
        util.display.display_visual_eval_images(sources,
                                                z_display=z_display,
                                                titles=titles,
                                                vmins=None,
                                                vmaxs=None,
                                                verbose=opts.verbose,
                                                path_z_ani=path_z_ani)
        if opts.save_images:
            path_base = os.path.join('test_output', os.path.basename(opts.load_path).split('.')[0])
            for idx, source in enumerate(sources):
                img = source[0, 0, ].astype(np.float32)
                path_img = os.path.join(path_base, 'img_{:02d}_{:s}.tif'.format(i, titles[idx]))
                util.save_img_np(img, path_img)
        if (opts.n_images is not None) and (i == (opts.n_images - 1)):
            break
    
def main():
    torch.cuda.set_device(opts.gpu_id)
    print('on GPU:', torch.cuda.current_device())
    
    if opts.use_train_set:
        print('*** Using training set ***')
    train_select = opts.use_train_set

    # load test dataset
    dataset = util.data.DataSet(path_load=opts.data_path,
                                train_select=train_select)
    print(dataset)

    # dims_chunk = (32, 208, 208)
    # dims_chunk = (48, 224, 320)
    # dims_pin = (0, 0, 0)
    # data_test = util.data.TestImgDataProvider(dataset, dims_chunk=dims_chunk, dims_pin=dims_pin)
    
    fixed_dim = (32, 224, 224)
    data_test = util.data.WholeImgDataProvider(dataset, fixed_dim)
    # data_test = util.data.WholeImgDataProvider(dataset, 'pad_mirror')
    
    # load model
    model = None
    if opts.load_path is not None:
        model = model_module.Model(load_path=opts.load_path)
    print(model)
    test_display(model, data_test)

if __name__ == '__main__':
    main()
