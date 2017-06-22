import argparse
import importlib
import gen_util
from util.SimpleLogger import SimpleLogger

parser = argparse.ArgumentParser()
parser.add_argument('--model_module', default='u_net_v0', help='name of the model module')
parser.add_argument('--save_dir', default='saved_models', help='save directory for trained model')
parser.add_argument('--dataProvider', default='DataProvider', help='name of Dataprovider class')
parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--n_batches_per_img', type=int, default=100, help='number batches to draw from each image')
parser.add_argument('--batch_size', type=int, default=64, help='size of each batch')

opts = parser.parse_args()
model_module = importlib.import_module('model_modules.'  + opts.model_module)

def train(model, data):
    """Here, an epoch is a training round in which every image file is used once."""
    file_list = ['some_file.czi']  # TODO: this should come from DataProvider
    n_optimizations = opts.n_epochs*len(file_list)*opts.n_batches_per_img
    print(n_optimizations, 'model optimizations expected')
    for epoch in range(opts.n_epochs):
        print('epoch:', epoch)
        for current_file in file_list:
            print('current file:', current_file)
            for batch_num in range(opts.n_batches_per_img):
                x, y = data.get_batch(16, dims_chunk=(32, 64, 64), dims_pin=(10, None, None))
                model.do_train_iter(x, y)

def test(model, data):
    # TODO: change data provider to pull from test image set
    x_test, y_true = data.get_batch(8, dims_chunk=(32, 64, 64), dims_pin=(10, None, None))
    y_pred = model.predict(x_test)
    gen_util.display_visual_eval_images(x_test, y_true, y_pred)

def main():
    print('main')
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

if __name__ == '__main__':
    main()

