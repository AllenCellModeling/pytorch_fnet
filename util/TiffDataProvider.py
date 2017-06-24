from . import DataProvider
import os
import glob
from aicsimage.io import omeTifReader
import numpy as np

class TiffDataProvider(DataProvider.DataProvider):
    def __init__(self, path_folders, n_epochs, n_batches_per_img, batch_size=64, rescale=None):
        """TODO: add description

        Each folder represents a set of TIFF images, two of which are the transmitted light
        channel and the DNA channel of a parent CZI file.

        Parameters:
        path_folders - file listing folders that contain TIFF images
        n_epochs - number of times to go through all folders
        n_batches_per_image - number of image chunks to pull from each image
        rescale - 3-element tuple describing any rescaling that should be applied to each image
        """
        if rescale is not None:
            raise NotImplementedError
        super().__init__()
        self.path_folders = path_folders
        self.n_epochs = n_epochs
        self.n_batches_per_img = n_batches_per_img

        # optional parameters
        self.batch_size = batch_size
        self.rescale = rescale
        
        with open(path_folders, 'r') as fin:  # create list of all folders
            self._folder_path_list = [name for name in fin.read().splitlines() if name]
        self._length = self.n_epochs*len(self._folder_path_list)*self.n_batches_per_img
        self._count_iter = 0
        self._count_epoch = 0
        self._idx_folder = 0
        self._count_batch = 0
        
        self._load_folder()  # load first folder with data

    def get_current_iter(self):
        return self._count_iter

    def get_current_epoch(self):
        return self._count_epoch
    
    def get_current_folder(self):
        return self._folder_path_list[self._idx_folder]

    def get_current_batch_num(self):
        return self._count_batch
    
    def _load_folder(self):
        """Populate vol_light_np, vol_nuc_np with data from files in folder."""
        def normalize_ar(ar):
            ar -= np.amin(ar)
            ar /= np.amax(ar)
            
        path = self._folder_path_list[self._idx_folder]
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
            tmp = fin.load()
            # print(type(tmp), tmp.shape, tmp.dtype)
            self.vol_dna_np = fin.load().astype(np.float32)[0, ]
            normalize_ar(self.vol_dna_np)

        self._idx_folder += 1
        if self._idx_folder % len(self._folder_path_list) == 0:
            self._idx_folder = 0
            self._count_epoch += 1

    def __len__(self):
        return self._length
    
    def _get_batch_and_incr(self):
        print('epoch: {:d} | folder: {:s} | batch_num: {:d}'.format(self._count_epoch,
                                                                    self._folder_path_list[self._idx_folder],
                                                                    self._count_batch))

        batch = self.get_batch()
        self._count_iter += 1
        self._count_batch += 1
        if self._count_batch == self.n_batches_per_img:
            self._count_batch = 0
            self._load_folder()
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        if self._count_iter == self._length:
            raise StopIteration
        return self._get_batch_and_incr()
        
if __name__ == '__main__':
    print('testing TiffDataProvder')
    tiff_dp = TiffDataProvider('some_folders.txt', 1, 3)
    print('len:', len(tiff_dp))
    count = 0
    for i in tiff_dp:
        count += 1
    print('count:', count)
        
