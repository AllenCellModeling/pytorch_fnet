import argparse
import importlib
import util.data
import util.data.transforms
import util.data.functions
import util.display
import numpy as np
import os
import torch.autograd
import torch.nn
import torch.nn.functional
import warnings
import subprocess
import pdb

warnings.filterwarnings('ignore', message='.*zoom().*')
warnings.filterwarnings('ignore', message='.*end of stream*')
warnings.filterwarnings('ignore', message='.*multiple of element size.*')

parser = argparse.ArgumentParser()
parser.add_argument('--path_data', default='data', help='path to data directory')
parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
parser.add_argument('--path_model', help='path to trained model')
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
    l1_losses, l2_losses = [], []
    test_or_train = 'test' if not opts.use_train_set else 'train'
    titles = ('signal', 'target', 'prediction')
    if opts.path_model is None:
        path_base = os.path.join('test_output', os.path.basename(opts.path_data).split('.')[0])
    else:
        path_base = os.path.join('test_output', os.path.basename(opts.path_model).split('.')[0])
    for i in indices:
        vmins, vmaxs = None, None
        try:
            x_test, y_true = data[i]
        except AttributeError:
            print('skipping....')
            continue
        if model is not None:
            y_pred = model.predict(x_test)
        else:
            y_pred = np.zeros(x_test.shape, dtype=np.float64)
        sources = (x_test[0, 0, ], y_true[0, 0, ], y_pred[0, 0, ])
        z_display = util.find_z_of_max_slice(y_true[0, 0, ])
        # z_display = util.find_z_max_std(x_test[0, 0, ]) - 10  # cells in focus (approximately) in bright-field image
        l1_loss, l2_loss = get_losses(y_pred, y_true)
        l1_losses.append(l1_loss)
        l2_losses.append(l2_loss)
        print('{:s} image: {:02d} | name: {} | z: {}'.format(test_or_train, i, data.get_name(i), z_display))
        print('L1 loss: {:.4f} | L2 loss: {:.4f}'.format(l1_loss, l2_loss))
        path_img_pre = os.path.join(path_base, 'img_{:s}_{:02d}'.format(test_or_train, i))
        if opts.save_images:
            for idx, source in enumerate(sources):
                img = source.astype(np.float32)
                path_img = path_img_pre + '_{:s}.tif'.format(titles[idx])
                util.save_img_np(img, path_img)
        path_z_ani = None
        z_save = None
        if opts.build_z_animation:
            path_z_ani = path_img_pre
            z_save = range(sources[0].shape[0])
            vmins = (-4, -0.5, 0)
            vmaxs = (4, 6, 3)
        util.display.display_eval_images(sources,
                                         z_display=z_display,
                                         titles=titles,
                                         vmins=vmins,
                                         vmaxs=vmaxs,
                                         path_save_dir=path_z_ani,
                                         z_save=z_save,
        )
        if opts.build_z_animation:
            path_gif = path_z_ani + '.gif'
            delay = 20
            cmd_str = 'convert -delay {} {}/*.png -trim +repage {}'.format(delay, path_z_ani, path_gif)
            subprocess.run(cmd_str, shell=True, check=True)
        if (opts.n_images is not None) and (i == (opts.n_images - 1)):
            break
    l1_loss_mean = np.mean(l1_losses)
    l2_loss_mean = np.mean(l2_losses)
    print('count: {:d} | L1 mean: {:.4f} | L2 mean: {:.4f}'.format(len(l1_losses), l1_loss_mean, l2_loss_mean))
    
def main():
    # load test dataset
    dataset = util.data.functions.load_dataset(opts.path_data)
    if opts.use_train_set:
        print('*** Using training set ***')
        dataset.use_train_set()
    else:
        dataset.use_test_set()
    # train_select = opts.use_train_set
    # dataset = util.data.DataSet2(path_model=opts.path_data,
    #                              train_select=train_select)
    print(dataset)

    if True:
        # dims_cropped = (32, 224, 224)
        dims_cropped = (32, '/16', '/16')
        cropper = util.data.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
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
    if opts.path_model is not None:
        model = model_module.Model(load_path=opts.path_model,
                                   gpu_ids=opts.gpu_ids
        )
    print('model:', model)
    test_display(model, data_test)

    

if __name__ == '__main__':
    main()
