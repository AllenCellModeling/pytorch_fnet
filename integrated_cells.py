import argparse
import tifffile
import numpy as np
import util
import util.data
import util.data.transforms
import model_modules.ttf_model
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID(s)')
opts = parser.parse_args()


path_dataset = 'data/dataset_saves/ttf_lamin_b1_latest.ds'
paths_models = (
    'saved_models/ttf_lamin_b1_latest.p',
    'saved_models/ttf_fibrillarin_latest.p',
    'saved_models/ttf_tom20_latest.p',
)

def make_example_tiff():
    img = np.zeros((51, 100, 200, 3), dtype=np.uint8)
    for z in range(img.shape[0]):
        color = z % 3
        print(z, color)
        img[z, :, :, color] = np.arange(200)
    tifffile.imsave('tmp_nometa.tif', img, photometric='rgb', metadata={'axes': 'ZYXC'})

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
    main()
