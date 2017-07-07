import argparse
import importlib
import util.data
import util.display
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', help='path to data directory')
parser.add_argument('--load_path', help='path to trained model')
parser.add_argument('--model_module', default='default_model', help='name of the model module')
parser.add_argument('--n_images', type=int, help='max number of images to test')
parser.add_argument('--test_mode', action='store_true', default=False, help='run test version of main')
parser.add_argument('--use_train_set', action='store_true', default=False, help='view predictions on training set images')
opts = parser.parse_args()

# command-line option imports
model_module = importlib.import_module('model_modules.'  + opts.model_module)

def test_display(model, data):
    y_pred = np.zeros((1, 1, 32, 64, 64))  # TODO replace with test chunk size
    for i, (x_test, y_true) in enumerate(data):
        if model is not None:
            y_pred = model.predict(x_test)
        util.display.display_visual_eval_images(x_test, y_true, y_pred)
        if i == (opts.n_images - 1):
            break

    return
    name_model = os.path.basename(opts.load_path).split('.')[0]

    # save predictions
    img_trans = x_test[0, 0, ].astype(np.float32)
    img_dna = y_true[0, 0, ].astype(np.float32)
    img_pred = y_pred[0, 0, ]
    
    name_pre = 'test_output/{:s}_'.format(name_model)
    name_post = '.tif'
    name_trans = name_pre + 'trans' + name_post
    name_dna = name_pre + 'dna' + name_post
    name_pred = name_pre + 'prediction' + name_post
    
    util.save_img_np(img_trans, name_trans)
    util.save_img_np(img_dna, name_dna)
    util.save_img_np(img_pred, name_pred)
    

def main_test_mode():
    test_set = ['data/3500000523_100X_20170314_D07_P27.czi']
    print(test_set)
    
    # aiming for 0.3 um/px
    # TODO: store this information in Model class
    z_fac = 0.97
    xy_fac = 0.36
    resize_factors = (z_fac, xy_fac, xy_fac)
    data_test = util.data.TiffCroppedDataProvider(test_set,
                                                  resize_factors=resize_factors,
                                                  shape_cropped=(32, 128, 128))
    # load model
    # model = model_module.Model(load_path=opts.load_path)
    model = None
    print(model)
    test_display(model, data_test)


def main():
    if opts.use_train_set:
        print('*** Using training set ***')
    train_select = opts.use_train_set

    # create test dataset
    dataset = util.data.DataSet(opts.data_path, train=train_select)
    print(dataset)

    dims_chunk = (32, 224, 224)
    data_test = util.data.TestImgDataProvider(dataset, dims_chunk=dims_chunk)
    
    # load model
    model = None
    if opts.load_path is not None:
        model = model_module.Model(load_path=opts.load_path)
    print(model)
    test_display(model, data_test)

if __name__ == '__main__':
    if opts.test_mode:
        main_test_mode()
    else:
        main()
