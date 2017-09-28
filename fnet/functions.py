import os
import numpy as np
from aicsimage.io import omeTifWriter
import pdb
import torch

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

def test_model(model, data, **kwargs):
    indices = range(len(data)) if kwargs.get('img_sel') is None else kwargs.get('img_sel')
    l1_losses, l2_losses = [], []
    l1_norm_losses, l2_norm_losses = [], []
    test_or_train = 'train' if data.using_train_set() else 'test'
    titles = ('signal', 'target', 'prediction')
    if kwargs.get('path_save') is not None:
        path_base = kwargs.get('path_save')
    elif kwargs.get('path_source') is not None and os.path.isdir(kwargs.get('path_source')):
        path_base = os.path.join(kwargs.get('path_source'), 'outputs')
    elif kwargs.get('path_model') is not None:
        path_base = os.path.join('test_output', os.path.basename(kwargs.get('path_model')).split('.')[0])
    elif kwargs.get('path_dataset') is not None:
        path_base = os.path.join('test_output', os.path.basename(kwargs.get('path_dataset')).split('.')[0])
    else:
        path_base = 'tmp'
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

        l1_loss, l2_loss = get_losses(y_pred, y_true)
        l1_norm_loss, l2_norm_loss = get_losses_norm(y_pred, y_true)
        l1_losses.append(l1_loss)
        l2_losses.append(l2_loss)
        l1_norm_losses.append(l1_norm_loss)
        l2_norm_losses.append(l2_norm_loss)
        print('{:s} image: {:02d} | name: {} | l1: {:.4f} | l2: {:.4f} | l1_norm: {:.4f} | l2_norm: {:.4f}'.format(
            test_or_train,
            i,
            data.get_name(i),
            l1_loss,
            l2_loss,
            l1_norm_loss,
            l2_norm_loss,
        ))
        path_img_pre = os.path.join(path_base, 'img_{:s}_{:02d}'.format(test_or_train, i))
        if kwargs.get('save_images'):
            for idx, source in enumerate(sources):
                img = source.astype(np.float32)
                path_img = path_img_pre + '_{:s}.tif'.format(titles[idx])
                save_img_np(img, path_img)
        if (kwargs.get('n_images') is not None) and (i >= (kwargs.get('n_images') - 1)):
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
    # l2_norm_argsort = np.argsort(l2_norm_losses)
    # print('Sorted by L2 norm loss')
    # for i in l2_norm_argsort:
    #     print('{:s} {:02d} | L2 loss: {:.4f}'.format(test_or_train, i, l2_norm_losses[i]))
    ret_dict = {
        'l1_' + test_or_train: l1_loss_mean,
        'l2_' + test_or_train: l2_loss_mean,
        'l1_norm_' + test_or_train: l1_norm_loss_mean,
        'l2_norm_' + test_or_train: l2_norm_loss_mean,
        'n_count_' + test_or_train: len(l1_losses),
    }
    return ret_dict
