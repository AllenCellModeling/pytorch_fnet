import datetime
import pytz
import os
import argparse
import importlib
import gen_util
import matplotlib.pyplot as plt
from util.SimpleLogger import SimpleLogger
import numpy as np
import torch
import util.TiffDataProvider


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='size of each batch')
parser.add_argument('--dataProvider', default='DataProvider', help='name of Dataprovider class')
parser.add_argument('--model_module', default='u_net_v0', help='name of the model module')
parser.add_argument('--n_batches_per_img', type=int, default=100, help='number batches to draw from each image')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--save_path', default='saved_models', help='save directory for trained model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--test_mode', action='store_true', default=False, help='run test version of main')

opts = parser.parse_args()
model_module = importlib.import_module('model_modules.'  + opts.model_module)

logger = SimpleLogger(('num_iter', 'epoch', 'file', 'batch_num', 'loss'),
                      'num_iter: %4d | epoch: %d | file: %s | batch_num: %3d | loss: %.4f')

def train(model, data):
    for batch in data:
        x, y = batch
        
        loss = model.do_train_iter(x, y)
        logger.add((data.get_current_iter(),
                    data.get_current_epoch(),
                    data.get_current_folder(),
                    data.get_current_batch_num(),
                    loss))

def test_whole(model, data):
    print('trying out whole image')
    # dimensions of image input into model must be powers of 2

    # # Pad original test image with zeros
    # shape = (1, 1) + data.vol_light_np.shape
    # shape_padded = (1, 1, 128, 256, 256)  # TODO: calculate automatically
    # offsets = [(shape_padded[i] - shape[i])//2 for i in range(5)]
    # slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(5)]
    # print(shape, shape_padded)
    # print(offsets)
    # print(slices)

    # Crop original image
    shape = (1, 1) + data.vol_light_np.shape
    shape_adj = (1, 1, 32, 128, 128)  # TODO: calculate automatically
    offsets = [(shape[i] - shape_adj[i])//2 for i in range(5)]
    slices = [slice(offsets[i], offsets[i] + shape_adj[i]) for i in range(5)]
    print(shape, shape_adj)
    print(offsets)
    print(slices)
    
    x_test = np.zeros(shape_adj)
    y_true = np.zeros(shape_adj)
    
    x_test[0, 0, :] = data.vol_light_np[slices[-3:]]
    y_true[0, 0, :] = data.vol_nuc_np[slices[-3:]]
    y_pred = model.predict(x_test)

    gen_util.display_visual_eval_images(x_test, y_true, y_pred)

    # save predictions
    img_light = x_test[0, 0, ].astype(np.float32)
    img_nuc = y_true[0, 0, ].astype(np.float32)
    img_pred = y_pred[0, 0, ]
    
    name_pre = 'test_output/{:s}_whole_cropped_'.format(model.meta['name'])
    name_post = '.tif'
    name_light = name_pre + 'light' + name_post
    name_nuc = name_pre + 'nuc' + name_post
    name_pred = name_pre + 'prediction' + name_post
    gen_util.save_img_np(img_light, name_light)
    gen_util.save_img_np(img_nuc, name_nuc)
    gen_util.save_img_np(img_pred, name_pred)
    
def test(model, data):
    # TODO: change data provider to pull from test image set
    x_test, y_true = data.get_batch(batch_size=8)
    y_pred = model.predict(x_test)
    gen_util.display_visual_eval_images(x_test, y_true, y_pred)

    return  # comment out to save data
    
    # save files
    n_examples = x_test.shape[0]
    
    # name_pre = 'test_output/iter_{:04d}_'.format(logger.log['num_iter'][-1])
    name_pre = 'test_output/{:s}_'.format(model.meta['name'])
    for i in range(n_examples):
        name_post = '_{:02d}.tif'.format(i)
        
        name_light = name_pre + 'light' + name_post
        name_nuc = name_pre + 'nuc' + name_post
        name_pred = name_pre + 'prediction' + name_post
        
        img_light = x_test[i, 0, ].astype(np.float32)
        img_nuc = y_true[i, 0, ].astype(np.float32)
        img_pred = y_pred[i, 0, ]

        gen_util.save_img_np(img_light, name_light)
        gen_util.save_img_np(img_nuc, name_nuc)
        gen_util.save_img_np(img_pred, name_pred)

def plot_logger_data():
    # TODO move to util
    x, y = logger.log['num_iter'], logger.log['loss']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes((1, 1, 1, .3), label='training loss')
    ax.plot(x, y)
    ax.set_xlabel('training iteration')
    ax.set_ylabel('MSE loss')
    plt.show()

def get_name_run():
    now_dt = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific'))
    name_run = now_dt.strftime('run_%y%m%d_%H%M%S')
    return name_run

def main_test():
    print('load model test')
    load_path = 'saved_models/run_170622_174530.p'
    model = model_module.Model(load_path=load_path)
    print(model)

    # load data. TODO: replace with DataProvider instance.
    fname = './test_images/20161209_C01_001.czi'
    # fname = './test_images/20161219_C01_034.czi'
    print('loading data:', fname)
    loader = gen_util.CziLoader(fname, channel_light=3, channel_nuclear=2)
    data = loader

    test_whole(model, data)

def main():
    # set seeds for randomizers
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    
    name_run = get_name_run()
    
    path_folders = 'some_folders.txt'
    data = util.TiffDataProvider.TiffDataProvider(path_folders, opts.n_epochs, opts.n_batches_per_img,
                                                  batch_size=opts.batch_size,
                                                  rescale=None)
    # instatiate model
    model = model_module.Model(mult_chan=32, depth=4)
    print(model)
    
    train(model, data)
    # test(model, data)
    # test_whole(model, data)
    
    # save model
    # model.save(os.path.join(opts.save_path, name_run + '.p'))
        
    # display logger data
    # logger.save_csv()
    # plot_logger_data()
    
def main_bk():
    # set seeds for randomizers
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    
    name_run = get_name_run()
    
    # load data. TODO: replace with DataProvider instance.
    fname = './test_images/20161209_C01_001.czi'
    loader = gen_util.CziLoader(fname, channel_light=3, channel_nuclear=2)
    data = loader
    
    # instatiate model
    model = model_module.Model(mult_chan=32, depth=4)
    print(model)
    
    train(model, data)
    # test(model, data)
    test_whole(model, data)
    
    # save model
    # model.save(os.path.join(opts.save_path, name_run + '.p'))
        
    # display logger data
    logger.save_csv()
    # plot_logger_data()

if __name__ == '__main__':
    if opts.test_mode:
        main_test()
    else:
        main()

