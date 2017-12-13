import os
import numpy as np
from aicsimage.io import omeTifWriter
import pdb
import torch
import json
import importlib

def find_z_max_intensity(ar):
    """Given a ZYX numpy array, return the z value of the XY-slice the highest total pixel intensity."""
    z_max = np.argmax(np.sum(ar, axis=(1, 2)))
    return int(z_max)

def find_z_max_std(ar):
    """Given a ZYX numpy array, return the z value of the XY-slice with the highest pixel intensity std."""
    z_max = np.argmax(np.std(ar, axis=(1, 2)))
    return int(z_max)

def get_vol_transformed(ar, transform):
    """Apply the transformation(s) to the supplied array and return the result."""
    if ar is None:
        return None
    result = ar
    if transform is None:
        pass
    elif isinstance(transform, (list, tuple)):
        for t in transform:
            result = t(result)
    else:
        result = transform(result)
    return result

def pad_mirror(ar, padding):
    """Pad 3d array using mirroring.

    Parameters:
    ar - (numpy.array) array to be padded
    padding - (tuple) per-dimension padding values
    """
    shape = tuple((ar.shape[i] + 2*padding[i]) for i in range(3))
    result = np.zeros(shape, dtype=ar.dtype)
    slices_center = tuple(slice(padding[i], padding[i] + ar.shape[i]) for i in range(3))
    result[slices_center] = ar
    # z-axis, centers
    if padding[0] > 0:
        result[0:padding[0], slices_center[1] , slices_center[2]] = np.flip(ar[0:padding[0], :, :], axis=0)
        result[ar.shape[0] + padding[0]:, slices_center[1] , slices_center[2]] = np.flip(ar[-padding[0]:, :, :], axis=0)
    # y-axis
    result[:, 0:padding[1], :] = np.flip(result[:, padding[1]:2*padding[1], :], axis=1)
    result[:, padding[1] + ar.shape[1]:, :] = np.flip(result[:, ar.shape[1]:ar.shape[1] + padding[1], :], axis=1)
    # x-axis
    result[:, :, 0:padding[2]] = np.flip(result[:, :, padding[2]:2*padding[2]], axis=2)
    result[:, :, padding[2] + ar.shape[2]:] = np.flip(result[:, :, ar.shape[2]:ar.shape[2] + padding[2]], axis=2)
    return result

def save_img_np(img_np, path):
    """Save image (numpy array, ZYX) as a TIFF."""
    path_dirname = os.path.dirname(path)
    if not os.path.exists(path_dirname):
        os.makedirs(path_dirname)
    with omeTifWriter.OmeTifWriter(path, overwrite_file=True) as fo:
        fo.save(img_np)
        print('saved tif:', path)

def save_run_state(path_save, loss_log):
    dict_state = dict(
        loss_log = loss_log
    )
    torch.save(dict_state, path_save)
    print('run state saved to:', path_save)

def load_model_from_dir(path_model_dir, gpu_ids=0):
    assert os.path.isdir(path_model_dir)
    path_model_state = os.path.join(path_model_dir, 'model.p')
    path_train_options = os.path.join(path_model_dir, 'train_options.json')
    with open(path_train_options, 'r') as fi:
        train_options = json.load(fi)
    model_module = importlib.import_module('model_modules.'  + train_options['model_module'])
    model = model_module.Model(
        gpu_ids=gpu_ids,
    )
    model.load_state(path_model_state)
    return model

def load_run_state(path_load):
    dict_state = torch.load(path_load)
    loss_log = dict_state['loss_log']
    print('run state loaded from:', path_load)
    return loss_log

def get_losses_norm(prediction, target):
    l1_norm = np.sum(np.absolute(target))
    l2_norm = np.sum(np.square(target))
    l1_loss_norm = np.sum(np.absolute(prediction - target))/l1_norm
    l2_loss_norm = np.sum(np.square(prediction - target))/l2_norm
    return l1_loss_norm, l2_loss_norm

def get_losses(prediction, target):
    y_pred = torch.autograd.Variable(torch.Tensor(prediction), volatile=True)
    y_targ = torch.autograd.Variable(torch.Tensor(target), volatile=True)
    l1_loss = torch.nn.L1Loss()
    l2_loss = torch.nn.MSELoss()
    l1_loss_float = float(l1_loss(y_pred, y_targ).data.numpy())
    l2_loss_float = float(l2_loss(y_pred, y_targ).data.numpy())
    return l1_loss_float, l2_loss_float

def test_model(
        model,
        dataset,
        img_sel = None,
        n_images = None,
        path_save_dir = None,
):
    if img_sel is not None:
        indices = [img_sel] if isinstance(img_sel, int) else img_sel
    else:
        indices = range(len(dataset))
    l1_losses, l2_losses = [], []
    l1_norm_losses, l2_norm_losses = [], []
    paths_czis, idx_list = [], []
    test_or_train = 'train' if dataset.using_train_set() else 'test'
    titles = ('signal', 'target', 'prediction')
    count = 0
    for i in indices:
        pair = dataset[i]
        if pair is None:
            print('skipping example', i)
            continue
        x_test, y_true = pair
        if model is not None:
            y_pred = model.predict(x_test)
        else:
            y_pred = np.zeros(x_test.shape, dtype=np.float64)
        sources = (x_test[0, 0, ], y_true[0, 0, ], y_pred[0, 0, ])

        l1_loss, l2_loss = get_losses(y_pred, y_true)
        l1_norm_loss, l2_norm_loss = get_losses_norm(y_pred, y_true)
        l1_losses.append(l1_loss)
        l2_losses.append(l2_loss)
        l1_norm_losses.append(l1_norm_loss)
        l2_norm_losses.append(l2_norm_loss)
        if path_save_dir is not None:
            path_img_pre = os.path.join(path_save_dir, 'img_{:s}_{:02d}'.format(test_or_train, i))
            for idx, source in enumerate(sources):
                img = source.astype(np.float32)
                path_img = path_img_pre + '_{:s}.tif'.format(titles[idx])
                save_img_np(img, path_img)
        paths_czis.append(dataset.get_name(i))
        idx_list.append(i)
        count += 1
        if n_images is not None and count >= n_images:
            break
    l1_loss_mean = np.mean(l1_losses)
    l2_loss_mean = np.mean(l2_losses)
    l1_norm_loss_mean = np.mean(l1_norm_losses)
    l2_norm_loss_mean = np.mean(l2_norm_losses)
    print('*****')
    print('count: {:d} | l1: {:.4f} | l2: {:.4f} | l1_norm: {:.4f} | l2_norm: {:.4f}'.format(
        len(l1_losses),
        l1_loss_mean,
        l2_loss_mean,
        l1_norm_loss_mean,
        l2_norm_loss_mean
    )
    )
    ret_dict = {
        'l1_' + test_or_train: l1_loss_mean,
        'l2_' + test_or_train: l2_loss_mean,
        'l1_norm_' + test_or_train: l1_norm_loss_mean,
        'l2_norm_' + test_or_train: l2_norm_loss_mean,
        'n_count_' + test_or_train: len(l1_losses),
    }
    ret_per_element = {
        'path_czi': paths_czis,
        'index': idx_list,
        'test_or_train': test_or_train,
        'l1': l1_losses,
        'l2': l2_losses,
        'l1_norm': l1_norm_losses,
        'l2_norm': l2_norm_losses,
    }
    return ret_dict, ret_per_element
