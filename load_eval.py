import argparse
import importlib
import util.data
import util.display
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', help='path to data directory')
parser.add_argument('--img_transform', default='normalize', help='transformation to apply to each image')
parser.add_argument('--load_path', help='path to trained model')
parser.add_argument('--model_module', default='default_model', help='name of the model module')
parser.add_argument('--n_images', type=int, help='max number of images to test')
parser.add_argument('--percent_test', type=float, default=0.1, help='percent of data to use for testing')
parser.add_argument('--test_mode', action='store_true', default=False, help='run test version of main')
parser.add_argument('--use_train_set', action='store_true', default=False, help='view predictions on training set images')
opts = parser.parse_args()

# command-line option imports
model_module = importlib.import_module('model_modules.'  + opts.model_module)

def test_cropped(model, data):
    # Crop original image

        # x_test[0, 0, :] = data.vol_trans_np[slices[-3:]]
        # y_true[0, 0, :] = data.vol_dna_np[slices[-3:]]
        # y_pred = model.predict(x_test)

    shape_example = (1, 1) + data.shape_cropped
    x_test = np.zeros(shape_example)
    y_true = np.zeros(shape_example)
    y_pred = np.zeros(shape_example)
    for i, pair_img_cropped in enumerate(data):
        x_test[0, 0, :] = pair_img_cropped[0]
        y_true[0, 0, :] = pair_img_cropped[1]
        if model is not None:
            y_pred = model.predict(x_test)
        util.display.display_visual_eval_images(x_test, y_true, y_pred)
    return

    # save predictions
    img_light = x_test[0, 0, ].astype(np.float32)
    img_nuc = y_true[0, 0, ].astype(np.float32)
    img_pred = y_pred[0, 0, ]
    
    name_pre = 'test_output/{:s}_whole_cropped_'.format(model.meta['name'])
    name_post = '.tif'
    name_light = name_pre + 'trans' + name_post
    name_nuc = name_pre + 'dna' + name_post
    name_pred = name_pre + 'prediction' + name_post
    # gen_util.save_img_np(img_light, name_light)
    # gen_util.save_img_np(img_nuc, name_nuc)
    # gen_util.save_img_np(img_pred, name_pred)
    

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
    test_cropped(model, data_test)


def main():
    # create test datasets
    dataset = util.data.DataSet(opts.data_path, percent_test=opts.percent_test)
    print(dataset)
    if not opts.use_train_set:
        img_set = dataset.get_test_set()
    else:
        print('*** Using training set ***')
        img_set = dataset.get_train_set()
        
    # aiming for 0.3 um/px
    # TODO: store this information in Model class
    z_fac = 0.97
    xy_fac = 0.36
    resize_factors = (z_fac, xy_fac, xy_fac)
    limit = len(img_set)
    if (opts.n_images is not None) and (opts.n_images < len(img_set)):
        limit = opts.n_images
    img_set = img_set[:limit]
    data_test = util.data.TiffCroppedDataProvider(img_set,
                                                  resize_factors=resize_factors,
                                                  shape_cropped=(32, 128, 128),
                                                  transform=opts.img_transform
    )
    # load model
    model = None
    if opts.load_path is not None:
        model = model_module.Model(load_path=opts.load_path)
    print(model)
    test_cropped(model, data_test)

if __name__ == '__main__':
    if opts.test_mode:
        main_test_mode()
    else:
        main()
