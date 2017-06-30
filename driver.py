import datetime
import pytz
import os
import argparse
import importlib
import gen_util
import matplotlib.pyplot as plt
from util.SimpleLogger import SimpleLogger
from util.DataSet import DataSet
from util.TiffDataProvider import TiffDataProvider
import numpy as np
import torch
import pdb
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=24, help='size of each batch')
parser.add_argument('--dataProvider', default='DataProvider', help='name of Dataprovider class')
parser.add_argument('--data_path', default='data', help='path to data directory')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--iter_save_log', type=int, default=250, help='iterations between log saves')
parser.add_argument('--iter_save_model', type=int, default=500, help='iterations between model saves')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--model_module', default='u_net_v0', help='name of the model module')
parser.add_argument('--n_batches_per_img', type=int, default=100, help='number batches to draw from each image')
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--no_model_save', action='store_true', help='do not save trained model')
parser.add_argument('--percent_test', type=float, default=0.1, help='percent of data to use for testing')
parser.add_argument('--run_name', help='name of run')
parser.add_argument('--save_path', default='saved_models', help='save directory for trained model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--show_plots', action='store_true', help='display plots and test images')
opts = parser.parse_args()

model_module = importlib.import_module('model_modules.'  + opts.model_module)

def train(model, data, logger):
    start = time.time()
    for i, batch in enumerate(data):
        x, y = batch
        # pdb.set_trace()
        loss = model.do_train_iter(x, y)
        stats = data.get_last_batch_stats()
        logger.add((
            stats['iteration'],
            stats['epoch'],
            stats['batch'],
            loss,
            stats['folder']
        ))
        if i % opts.iter_save_log == 0:
            logger.save_csv()
        if i % opts.iter_save_model == 0 and i > 0 and not opts.no_model_save:
            model.save(os.path.join(opts.save_path, logger.logger_name + '.p'))
            
    t_elapsed = time.time() - start
    print('***** Training Time *****')
    print('total:', t_elapsed)
    print('per epoch:', t_elapsed/opts.n_epochs)
    print()
    
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

def get_run_name():
    now_dt = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific'))
    run_name = now_dt.strftime('run_%y%m%d_%H%M%S')
    return run_name

def main():
    # set seeds for randomizers
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    torch.cuda.set_device(opts.gpu_id)
    print('on GPU:', torch.cuda.current_device())

    run_name = opts.run_name
    if run_name is None:
        run_name = get_run_name()
    logger = SimpleLogger(('num_iter', 'epoch', 'batch_num', 'loss', 'file'),
                          'num_iter: %4d | epoch: %d | batch_num: %3d | loss: %.6f | file: %s',
                          logger_name=run_name)

    # create train, test datasets
    dataset = DataSet(opts.data_path, percent_test=opts.percent_test)
    print(dataset)
    test_set = dataset.get_test_set()
    train_set = dataset.get_train_set()
    
    # aiming for 0.3 um/px
    z_fac = 0.97
    xy_fac = 0.36
    resize_factors = (z_fac, xy_fac, xy_fac)
    data_train = TiffDataProvider(train_set, opts.n_epochs, opts.n_batches_per_img,
                                  batch_size=opts.batch_size,
                                  resize_factors=resize_factors)
    data_test = TiffDataProvider(test_set, 1, 8,
                                 batch_size=1,
                                 resize_factors=resize_factors)
    
    # instatiate model
    model = model_module.Model(mult_chan=32, depth=4, lr=opts.lr)
    print(model)
    train(model, data_train, logger)
        
    # display logger data
    logger.save_csv()
    
    # save model
    if not opts.no_model_save:
        model.save(os.path.join(opts.save_path, run_name + '.p'))
        
    if opts.show_plots:
        plot_logger_data()
    
if __name__ == '__main__':
    main()
