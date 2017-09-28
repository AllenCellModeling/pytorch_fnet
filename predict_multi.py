import argparse
import tifffile
import numpy as np
import fnet
import fnet.data
import fnet.data.transforms
import fnet.data.functions
import model_modules.ttf_model
import os
import subprocess
import pdb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

palette_seaborn_colorblind = ((142, 255),
                              (115, 255),
                              (18, 255),
                              (231, 103),
                              (39, 184),
                              (142, 160),
)

parser = argparse.ArgumentParser()
parser.add_argument('--path_source', help='path to data CZI or saved dataset')
parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID(s)')
parser.add_argument('--save_rgb', action='store_true', help='save rgb images')
parser.add_argument('--img_sel', type=int, nargs='*', help='select images to test')
opts = parser.parse_args()

paths_models = (
    'saved_models/ttf_lamin_b1_latest.p',
    'saved_models/ttf_fibrillarin_latest.p',
    'saved_models/ttf_tom20_latest.p',
    'saved_models/ttf_sec61_beta_latest.p',
    'saved_models/ttf_bf_dna_no_relu/model.p',
)
fname_tags = ('signal', 'target', 'lamin_b1' ,'fibrillarin' ,'tom20', 'sec61', 'dna')

def convert_ar_grayscale_to_rgb(ar, hue_sat):
    """Converts grayscale image to RGB uint8 image.
    ar - numpy.ndarray representing grayscale image
    hue_sat - 2-element tuple representing a color's hue and saturation values.
              Elements should be [0.0, 1.0] if floats, [0, 255] if ints.
    """
    hue = hue_sat[0]/256.0 if isinstance(hue_sat[0], int) else hue_sat[0]
    sat = hue_sat[1]/256.0 if isinstance(hue_sat[1], int) else hue_sat[1]
    ar_float = ar.astype(np.float) - np.min(ar).astype(np.float)  # TODO: can possibly overflow
    ar_float /= np.max(ar_float)
    ar_hsv = np.zeros(ar.shape + (3, ), dtype=np.float)
    ar_hsv[..., 0] = hue
    ar_hsv[..., 1] = sat
    ar_hsv[..., 2] = ar_float
    ar_rgb = matplotlib.colors.hsv_to_rgb(ar_hsv)
    ar_rgb *= 256.0
    ar_rgb[ar_rgb == 256.0] = 255.0
    return ar_rgb.astype(np.uint8)

def blend_ar(ars, weights):
    shape_exp = ars[0].shape
    ar_all = np.zeros(ars[0].shape, dtype=np.float)
    for i in range(len(ars)):
        assert weights[i] >= 0.0
        assert ars[i].shape == shape_exp
        ar_all += weights[i]*ars[i]
    ar_all -= np.min(ar_all)
    ar_all /= np.max(ar_all)
    ar_all *= 256.0
    ar_all[ar_all == 256.0] = 255.0
    return ar_all.astype(np.uint8)

def make_example_tiff():
    hue_sats = palette_seaborn_colorblind[3:]
    img_r_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    img_g_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    img_b_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    for z in range(img_r_pre.shape[0]):
        img_r_pre[z, :20, :50] = 255 - 2*z
        img_g_pre[z, 40:60, 50:100] = 100 + 2*z
        img_b_pre[z, 65:85, 150:200] = 150 + 2*z
    path_base = 'test_output'
    img_r = convert_ar_grayscale_to_rgb(img_r_pre, hue_sats[0])
    img_g = convert_ar_grayscale_to_rgb(img_g_pre, hue_sats[1])
    img_b = convert_ar_grayscale_to_rgb(img_b_pre, hue_sats[2])
    tifffile.imsave(os.path.join(path_base, 'r.tif'), img_r, photometric='rgb')
    tifffile.imsave(os.path.join(path_base, 'g.tif'), img_g, photometric='rgb')
    tifffile.imsave(os.path.join(path_base, 'b.tif'), img_b, photometric='rgb')
    img_all = blend_ar((img_r, img_g, img_b), (1/3, 1/3, 1/3))
    tifffile.imsave(os.path.join(path_base, 'all.tif'), img_all, photometric='rgb')

class GhettoIntegratedCells(object):
    def __init__(self, paths_models):
        self._paths_models = paths_models
        self._verify()

    def get_predictions(self, x_signal):
        assert isinstance(x_signal, np.ndarray)
        assert x_signal.ndim == 5
        predictions = []
        for path in self._paths_models:
            print(path)
            if path is None:
                prediction = np.zeros((x_signal.shape), dtype=x_signal.dtype)
            else:
                model = model_modules.ttf_model.Model(load_path=path,
                                                      gpu_ids=opts.gpu_ids)
                prediction = model.predict(x_signal)
            predictions.append(prediction)
        return predictions

    def _verify(self):
        for path in self._paths_models:
            if path is not None:
                assert os.path.isfile(path)

def display_gic_slice(sources, z_display, titles=None,
                      path_save_dir=None):
    """
    sources - iterable of numpy.ndarrays
    z_display - z-value or iterable of z-values of slice(s) to display
    """
    def get_fig_axes_layout_0():
        fig = plt.figure(figsize=(5, 6), dpi=180)
        gs = gridspec.GridSpec(3, 4, hspace=0.2, wspace=0.05, left=0.0, right=1.0)
        axes = []
        axes.append(fig.add_subplot(gs[1, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))
        axes.append(fig.add_subplot(gs[1, 1]))
        axes.append(fig.add_subplot(gs[2, 1]))
        axes.append(fig.add_subplot(gs[1, 2]))
        return fig, axes
    
    def get_fig_axes_layout_1():
        fig = plt.figure(dpi=200)
        gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.05, left=0.0, right=1.0)
        axes = []
        axes.append(fig.add_subplot(gs[0, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))
        axes.append(fig.add_subplot(gs[1, 0]))
        axes.append(fig.add_subplot(gs[1, 1]))
        axes.append(fig.add_subplot(gs[:, 2:]))
        return fig, axes

    def get_fig_axes_layout_2():
        fig = plt.figure(figsize=(5, 6), dpi=180)
        gs = gridspec.GridSpec(3, 4, hspace=0.2, wspace=0.05, left=0.0, right=1.0)
        axes = []
        axes.append(fig.add_subplot(gs[0, 1]))
        axes.append(fig.add_subplot(gs[1, 0]))
        axes.append(fig.add_subplot(gs[1, 1]))
        axes.append(fig.add_subplot(gs[1, 2]))
        axes.append(fig.add_subplot(gs[2, 1]))
        return fig, axes
    
    def get_fig_axes_layout_3():
        fig = plt.figure(figsize=(6, 2), dpi=400)
        gs = gridspec.GridSpec(1, 5, wspace=0.05, left=0.0, right=1.0)
        axes = []
        axes.append(fig.add_subplot(gs[0, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))
        axes.append(fig.add_subplot(gs[0, 2]))
        axes.append(fig.add_subplot(gs[0, 3]))
        axes.append(fig.add_subplot(gs[0, 4]))
        return fig, axes
    
    if isinstance(z_display, int):
        z_iter = (z_display, )
    else:
        z_iter = z_display
    if path_save_dir is not None:
        if not os.path.exists(path_save_dir):
            os.makedirs(path_save_dir)
    for z_val in z_iter:
        fig, axes = get_fig_axes_layout_3()
        for i, ax in enumerate(axes):
            if titles is not None:
                ax.set_title(titles[i], loc='left', fontsize='small')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            img = sources[i][z_val, ...]
            kwargs = {'interpolation':'bilinear'}
            if i == 0:
                kwargs['cmap'] = 'gray'
                kwargs['vmin'] = -5
                kwargs['vmax'] = 5
            else:
                kwargs['vmin'] = 0
                kwargs['vmax'] = 255
            ax.imshow(img, **kwargs)
        plt.show()
        if path_save_dir is not None:
            path_save = os.path.join(path_save_dir, 'z_{:02d}.png'.format(z_val))
            fig.savefig(path_save)
            print('saved:', path_save)
        plt.close(fig)

def get_sources_from_files(path):
    def order(fname):
        if 'bf' in fname: return 0
        if 'lamin' in fname: return 1
        if 'fibrillarin' in fname: return 2
        if 'tom' in fname: return 3
        if 'all' in fname: return 10
        return 0
    files = [i.path for i in os.scandir(path) if i.is_file()]
    paths_sources = [i for i in files if (('rgb.tif' in i) or ('bf.tif' in i))]
    paths_sources.sort(key=order)
    sources = []
    for path in paths_sources:
        source = tifffile.imread(path)
        sources.append(source)
    return sources
    
def set_plot_style():
    plt.rc('axes', titlepad=4)

def get_dataset(path_source):
    if path_source.lower().endswith('.czi'):
        # aiming for 0.3 um/px
        z_fac = 0.97
        xy_fac = 0.217  # timelapse czis; original scale 0.065 um/px
        resize_factors = (z_fac, xy_fac, xy_fac)
        resizer = fnet.data.transforms.Resizer(resize_factors)
        transforms = ((resizer, fnet.data.transforms.sub_mean_norm),
                      (resizer, fnet.data.transforms.sub_mean_norm))
        dataset = fnet.data.functions.create_dataset(
            path_source=path_source,
            name_signal='bf',
            name_target='struct',
            train_split=0,
            transforms=transforms,
        )
    else:
        dataset = fnet.data.functions.load_dataset(path_source)
    dataset.use_test_set()
    return dataset

    
def predict_multi_model(dataset, indices, path_base_save='test_output/gic/tmp', save_rgb=False):
    dims_cropped = (32, '/16', '/16')
    cropper = fnet.data.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
    transforms = (cropper, cropper)
    data_test = fnet.data.TestImgDataProvider(dataset, transforms)
    
    gic = GhettoIntegratedCells(paths_models)
    
    if not os.path.exists(path_base_save):
        os.makedirs(path_base_save)

    # color/title setup
    n_models = len(paths_models)
    
    for ind in indices:
        signal, target = data_test[ind]
        predictions = gic.get_predictions(signal)
        sources = [signal, target] + predictions
        for i, pred in enumerate(sources):
            img = pred[0, 0, ].astype(np.float32)
            path_img = os.path.join(path_base_save, 'img_{:04d}_{:s}_gray.tif'.format(ind, fname_tags[i]))
            tifffile.imsave(path_img, img, photometric='minisblack')

def make_gif(path_source,
             timelapse=False,
             delay=20,
             z_slice=13,
):
    """
    path_source - directory of source file/directory
    timelapse - (bool) 
    z_slice - (int) only used for timelapse gifs
    """
    if timelapse:
        assert os.path.isdir(path_source)
        path_source_base = os.path.basename(path_source)
        source_str = '"{:s}*_all_rgb.tif[{:d}]"'.format(
            path_source + os.path.sep,
            z_slice
        )
        dst_str = '{:s}gic_{:s}_z{:02d}.gif'.format(
            path_source + os.path.sep,
            path_source_base,
            z_slice
        )
        cmd_str = 'convert -delay {:d} {:s} {:s}'.format(
            delay,
            source_str,
            dst_str
        )
        print(cmd_str)
    else:
        raise NotImplementedError
    subprocess.run(cmd_str, shell=True, check=True)

if __name__ == '__main__':
    dataset = get_dataset(opts.path_source)
    print(dataset)

    path_dataset_base = os.path.basename(opts.path_source)
    path_out = os.path.join('test_output', 'gic', path_dataset_base)
    indices = range(len(dataset)) if opts.img_sel is None else opts.img_sel
    predict_multi_model(dataset, indices, path_out, save_rgb=opts.save_rgb)
    # predict_multi_model(dataset, indices, path_out, save_rgb=opts.save_rgb)

    # make_gif(path_source=path_out,
    #          timelapse=dataset.is_timelapse(),
    #          z_slice=13,
    # )
    
    # sources = get_sources_from_files(path_source)
    # set_plot_style()
    # display_gic_slice(sources, range(32),
    #                   titles=('bright-field', 'Lamin B1', 'Fibrillarin', 'Tom20', 'all'),
    #                   path_save_dir=path_save_dir
    # )
