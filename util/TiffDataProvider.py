from . import DataProvider
# import DataProvider
import os
import glob
from aicsimage.io import omeTifReader
import numpy as np

class TiffDataProvider(DataProvider.DataProvider):
    def __init__(self, folder_list, n_epochs, n_batches_per_img, batch_size=16, resize_factors=None):
        """TODO: add description

        Each folder represents a set of TIFF images, two of which are the transmitted light
        channel and the DNA channel of a parent CZI file.

        Parameters:
        path_folders - file listing folders that contain TIFF images
        n_epochs - number of times to go through all folders
        n_batches_per_image - number of image chunks to pull from each image
        resize - 3-element tuple describing any rescaling that should be applied to each image
        """
        super().__init__()
        self._folder_list = folder_list
        self._n_epochs = n_epochs
        self._n_batches_per_img = n_batches_per_img
        self._resize_factors = resize_factors
        self.batch_size = batch_size  # used in super class
        
        self._length = self._n_epochs*len(self._folder_list)*self._n_batches_per_img
        
        self._last_batch_stats = {}
        
        self._count_iter = 0
        self._count_epoch = 0
        self._idx_folder = 0
        self._count_batch = 0
        self._active_folder = None

    def get_last_batch_stats(self):
        return self._last_batch_stats
    
    def _load_folder(self):
        """Populate vol_light_np, vol_nuc_np with data from files in folder."""
        def normalize_ar(ar):
            ar -= np.amin(ar)
            ar /= np.amax(ar)

        path = self._folder_list[self._idx_folder]
        print('TiffDataProvider: loading', path)
        trans_fname_list = glob.glob(os.path.join(path, '*_trans.tif'))
        dna_fname_list = glob.glob(os.path.join(path, '*_dna.tif'))
        if len(trans_fname_list) > 1:
            print('WARNING: more than 1 transmitted light image in:', path)
        if len(dna_fname_list) > 1:
            print('WARNING: more than 1 DNA image in:', path)
        path_trans = trans_fname_list[0]
        path_dna = dna_fname_list[0]
        with omeTifReader.OmeTifReader(path_trans) as fin:
            self.vol_trans_np = fin.load().astype(np.float32)[0, ]
            normalize_ar(self.vol_trans_np)
        with omeTifReader.OmeTifReader(path_dna) as fin:
            self.vol_dna_np = fin.load().astype(np.float32)[0, ]
            normalize_ar(self.vol_dna_np)
        if self._resize_factors is not None:
            self.resize_data(self._resize_factors)
        
        self._active_folder = self._folder_list[self._idx_folder]
        self._last_batch_stats['folder'] = self._active_folder
        self._last_batch_stats['epoch'] = self._count_epoch
        self._idx_folder += 1
        if self._idx_folder % len(self._folder_list) == 0:
            self._idx_folder = 0
            self._count_epoch += 1

    def _incr_stuff(self):
        self._last_batch_stats['iteration'] = self._count_iter
        self._last_batch_stats['batch'] = self._count_batch
        self._count_iter += 1
        self._count_batch += 1
        if self._count_batch == self._n_batches_per_img:
            self._count_batch = 0
            self._last_batch_stats['folder'] = self._active_folder
            self._active_folder = None

    def __len__(self):
        return self._length
    
    def __iter__(self):
        return self

    def __next__(self):
        if self._count_iter == self._length:
            raise StopIteration
        if self._active_folder is None:
            self._load_folder()
        batch = self.get_batch()
        self._incr_stuff()
        return batch

def test():
    print('testing TiffDataProvder')
    data_path = '../data_test'
    data_set = [i.path for i in os.scandir(data_path) if i.is_dir()]
    print(data_set)
    tiff_dp = TiffDataProvider(data_set, 3, 2)
    for batch in tiff_dp:
        stats = tiff_dp.get_last_batch_stats()
        print(stats)

        
if __name__ == '__main__':
    test()
        
