import argparse
import importlib
import gen_util
import matplotlib.pyplot as plt
from util.SimpleLogger import SimpleLogger
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_module', default='u_net_v0', help='name of the model module')
parser.add_argument('--save_dir', default='saved_models', help='save directory for trained model')
parser.add_argument('--dataProvider', default='DataProvider', help='name of Dataprovider class')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--n_batches_per_img', type=int, default=100, help='number batches to draw from each image')
parser.add_argument('--batch_size', type=int, default=64, help='size of each batch')

opts = parser.parse_args()
model_module = importlib.import_module('model_modules.'  + opts.model_module)

logger = SimpleLogger(('num_iter', 'epoch', 'file', 'batch_num', 'loss'),
                      'num_iter: %4d | epoch: %d | file: %s | batch_num: %3d | loss: %.4f')

def train(model, data):
    """Here, an epoch is a training round in which every image file is used once."""
    file_list = ['some_file.czi']  # TODO: this should come from DataProvider
    n_optimizations = opts.n_epochs*len(file_list)*opts.n_batches_per_img
    print(n_optimizations, 'model optimizations expected')
    num_iter = 0  # track total training iterations for this model
    for epoch in range(opts.n_epochs):
        print('epoch:', epoch)
        for fname in file_list:
            print('current file:', fname)
            for batch_num in range(opts.n_batches_per_img):
                x, y = data.get_batch(16, dims_chunk=(32, 64, 64), dims_pin=(10, None, None))
                loss = model.do_train_iter(x, y)
                logger.add((num_iter, epoch, fname, batch_num, loss))
                num_iter += 1

def test(model, data):
    # TODO: change data provider to pull from test image set
    x_test, y_true = data.get_batch(8, dims_chunk=(32, 64, 64), dims_pin=(10, None, None))
    y_pred = model.predict(x_test)
    gen_util.display_visual_eval_images(x_test, y_true, y_pred)

    # save files
    n_examples = x_test.shape[0]
    num_iter = logger.log['num_iter'][-1]
    name_pre = 'test_output/iter_{:04d}_'.format(num_iter)
    for i in range(n_examples):
        name_post = '_{:02d}.tif'.format(i)
        
        name_light = name_pre + 'light' + name_post
        name_nuc = name_pre + 'nuc' + name_post
        name_pred = name_pre + 'prediction' + name_post
        
        img_light = x_test[i, 0, ].astype(np.float32)
        img_nuc = y_true[i, 0, ].astype(np.float32)
        img_pred = y_pred[i, 0, ]

        if False:
            gen_util.save_img_np(img_light, name_light)
            gen_util.save_img_np(img_nuc, name_nuc)
            gen_util.save_img_np(img_pred, name_pred)

def plot_logger_data():
    x, y = logger.log['num_iter'], logger.log['loss']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes((1, 1, 1, .3), label='training loss')
    ax.plot(x, y)
    ax.set_xlabel('training iteration')
    ax.set_ylabel('MSE loss')
    plt.show()

def main():
    # load data. TODO: replace with DataProvider instance.
    fname = './test_images/20161209_C01_001.czi'
    loader = gen_util.CziLoader(fname, channel_light=3, channel_nuclear=2)
    data = loader
    
    
    # instatiate model
    model = model_module.Model(mult_chan=32, depth=4)
    print(model)
    
    # train model
    train(model, data)
    
    # save model
    # test model
    test(model, data)

    # display logger data
    # logger.save_csv()
    plot_logger_data()

if __name__ == '__main__':
    main()

