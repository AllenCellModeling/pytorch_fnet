from util.data.dataset import DataSet
from util import get_vol_transformed
from util.io import read_tifs

class DataSetCellnucNotCellnuc(DataSet):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
        
    def __getitem__(self, index):
        """Returns arrays corresponding to the transmitted light and DNA channels of the folder specified by index.

        Once the files are read in as numpy arrays, apply the transformations specified in the constructor.

        Returns:
        volumes - 4-element tuple or None. If the file read was successful, return tuple
        of volumes in order (dna, ) else return None
        """
        path_folder = self._active_set[index]
        volumes_pre = read_tifs(path_folder, ('_dna.tif', '_nuc.tif', '_cell.tif'))
        if volumes_pre is None:
            return None
        volumes = (get_vol_transformed(volumes_pre[0], self._transform),
                   get_vol_transformed(volumes_pre[1], self._target_transform),
                   get_vol_transformed(volumes_pre[2], self._target_transform))
        # print('DEBUG: DataSet. item shapes:', volumes[0].shape, volumes[1].shape)
        return volumes
