import argparse
import tifffile
import numpy as np
# import util
# import util.data
# import util.data.transforms
import model_modules.ttf_model
import os
import pdb
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID(s)')
opts = parser.parse_args()


path_dataset = 'data/dataset_saves/ttf_lamin_b1_latest.ds'
paths_models = (
    'saved_models/ttf_lamin_b1_latest.p',
    'saved_models/ttf_fibrillarin_latest.p',
    'saved_models/ttf_tom20_latest.p',
)

def convert_ar_grayscale_to_rgb(ar, hue):
    """Converts grayscale image to RGB uint8 image."""
    # shift and scale input array
    ar_float = ar.astype(np.float) - np.min(ar).astype(np.float)  # TODO: can possibly overflow
    ar_float /= np.max(ar_float)
    ar_hsv = np.zeros(ar.shape + (3, ), dtype=np.float)
    ar_hsv[..., 0] = hue
    ar_hsv[..., 1] = 1.0
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
    ar_all /= np.max(ar_all)
    ar_all *= 256.0
    ar_all[ar_all == 256.0] = 255.0
    return ar_all.astype(np.uint8)

def make_example_tiff():
    hues = np.random.rand(3)
    img_r_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    img_g_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    img_b_pre = np.zeros((51, 100, 200), dtype=np.uint8)
    for z in range(img_r_pre.shape[0]):
        img_r_pre[z, :20, :50] = 255 - 2*z
        img_g_pre[z, 40:60, 50:100] = 100 + 2*z
        img_b_pre[z, 65:85, 150:200] = 150 + 2*z
    path_base = 'test_output'
    # img_r = convert_ar_grayscale_to_rgb(img_r_pre, 0)
    # img_g = convert_ar_grayscale_to_rgb(img_g_pre, 1/3)
    # img_b = convert_ar_grayscale_to_rgb(img_b_pre, 2/3)
    img_r = convert_ar_grayscale_to_rgb(img_r_pre, hues[0])
    img_g = convert_ar_grayscale_to_rgb(img_g_pre, hues[1])
    img_b = convert_ar_grayscale_to_rgb(img_b_pre, hues[2])
    tifffile.imsave(os.path.join(path_base, 'r.tif'), img_r, photometric='rgb', metadata={'axes': 'ZYXC'})
    tifffile.imsave(os.path.join(path_base, 'g.tif'), img_g, photometric='rgb', metadata={'axes': 'ZYXC'})
    tifffile.imsave(os.path.join(path_base, 'b.tif'), img_b, photometric='rgb', metadata={'axes': 'ZYXC'})
    img_all = blend_ar((img_r, img_g, img_b), (1/3, 1/3, 1/3))
    tifffile.imsave(os.path.join(path_base, 'all.tif'), img_all, photometric='rgb', metadata={'axes': 'ZYXC'})

class GhettoIntegratedCells(object):
    def __init__(self, paths_models):
        self._paths_models = paths_models
        self._verify()

    def get_predictions(self, x_signal):
        assert isinstance(x_signal, np.ndarray)
        assert len(x_signal.shape) == 5
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
            
def main():
    # load test dataset
    path_load = path_dataset
    train_select = False
    dataset = util.data.DataSet2(path_load=path_load,
                                 train_select=train_select)
    print(dataset)
    
    dims_cropped = (32, '/16', '/16')
    cropper = util.data.transforms.Cropper(dims_cropped, offsets=('mid', 0, 0))
    transforms = (cropper, )
    data_test = util.data.TestImgDataProvider(dataset, transforms)
    x_signal = data_test.get_item_sel(3, 0)

    gic = GhettoIntegratedCells(paths_models)
    predictions = gic.get_predictions(x_signal)
    sources = [x_signal] + predictions
    titles = ('bf', 'lamin_b1' ,'fibrillarin' ,'tom_20')
    for i, pred in enumerate(sources):
        img = pred[0, 0, ].astype(np.float32)
        path_img = os.path.join('test_output', 'gic', 'img_{:02d}_{:s}.tif'.format(0, titles[i]))
        util.save_img_np(img, path_img)

if __name__ == '__main__':
    make_example_tiff()
    # main()
