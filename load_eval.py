import argparse
import importlib
import util.data
import util.data.transforms
import util.display
import numpy as np
import os
import torch.autograd
import torch.nn
import torch.nn.functional
import warnings
import pdb

warnings.filterwarnings('ignore', message='.*zoom().*')
warnings.filterwarnings('ignore', message='.*end of stream*')
warnings.filterwarnings('ignore', message='.*multiple of element size.*')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='data', help='path to data directory')
parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
parser.add_argument('--load_path', help='path to trained model')
parser.add_argument('--model_module', default='ttf_model', help='name of the model module')
parser.add_argument('--n_images', type=int, help='max number of images to test')
parser.add_argument('--img_sel', type=int, nargs='*', help='select images to test')
parser.add_argument('--build_z_animation', action='store_true', default=False, help='save each z slice of test images')
parser.add_argument('--save_images', action='store_true', default=False, help='save test image results')
parser.add_argument('--use_train_set', action='store_true', default=False, help='view predictions on training set images')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='enable verbose output')
opts = parser.parse_args()

model_module = importlib.import_module('model_modules.'  + opts.model_module)

def get_losses(prediction, target):
    y_pred = torch.autograd.Variable(torch.Tensor(prediction), volatile=True)
    y_targ = torch.autograd.Variable(torch.Tensor(target), volatile=True)
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    l1_loss_float = float(l1_loss(y_pred, y_targ).data.numpy())
    l2_loss_float = float(l2_loss(y_pred, y_targ).data.numpy())
    return l1_loss_float, l2_loss_float

def test_display(model, data):
    indices = range(len(data)) if opts.img_sel is None else opts.img_sel
    losses = []
    for i in indices:
        try:
            x_test, y_true = data[i]
        except AttributeError:
            print('skipping....')
            continue
        if model is not None:
            y_pred = model.predict(x_test)
        else:
            y_pred = np.zeros(x_test.shape, dtype=np.float32)
        path_z_ani = None
        if opts.build_z_animation:
            path_z_ani = 'presentation/' + ('test' if not opts.use_train_set else 'train') + '_{:02d}'.format(i)
        sources = (x_test, y_true, y_pred)
        # z_display = util.find_z_of_max_slice(y_true[0, 0, ]) + 3
        z_display = util.find_z_max_std(x_test[0, 0, ]) - 10  # cells in focus (approximately) in bright-field image
        if z_display < 0:
            z_display = 15
        titles = ('signal', 'target', 'prediction')
        print('test image: {:02d} | name: {} | z: {}'.format(i, data.get_name(i), z_display))
        losses.append(get_losses(y_pred, y_true))
        print('L1 loss: {:.4f} | L2 loss: {:.4f}'.format(losses[-1][0], losses[-1][1]))
        util.display.display_visual_eval_images(sources,
                                                z_display=z_display,
                                                titles=titles,
                                                vmins=None,
                                                vmaxs=None,
                                                verbose=opts.verbose,
                                                path_z_ani=path_z_ani)
        if opts.save_images:
            if opts.load_path is None:
                path_base = os.path.join('test_output', os.path.basename(opts.data_path).split('.')[0])
            else:
                path_base = os.path.join('test_output', os.path.basename(opts.load_path).split('.')[0])
            for idx, source in enumerate(sources):
                img = source[0, 0, ].astype(np.float32)
                path_img = os.path.join(path_base, 'img_{:02d}_{:s}.tif'.format(i, titles[idx]))
                util.save_img_np(img, path_img)
        if (opts.n_images is not None) and (i == (opts.n_images - 1)):
            break
    
def main():
    if opts.use_train_set:
        print('*** Using training set ***')
    train_select = opts.use_train_set

    # load test dataset
    dataset = util.data.DataSet2(path_load=opts.data_path,
                                 train_select=train_select)
    print(dataset)

    if True:
        # dims_cropped = (32, 224, 224)
        dims_cropped = (32, '/16', '/16')
        cropper = util.data.transforms.Cropper(dims_cropped, offsets=(16, 0, 0))
        transforms = (cropper, cropper)
    else:  # for models with no padding
        # fixed_dim = (28, 220, 220)
        # fixed_dim = (36, 220, 220)
        fixed_dim = (44, 220, 220)
        cropper = util.data.transforms.Cropper(fixed_dim, offsets=(24 - fixed_dim[0]//2, 0, 0))
        padder =  util.data.transforms.ReflectionPadder3d((28, 28, 28))
        transforms = ((cropper, padder), (cropper))
    
    data_test = util.data.TestImgDataProvider(dataset, transforms)
    
    # load model
    model = None
    if opts.load_path is not None:
        model = model_module.Model(load_path=opts.load_path,
                                   gpu_ids=opts.gpu_ids
        )
    print(model)
    test_display(model, data_test)

if __name__ == '__main__':
    main()
