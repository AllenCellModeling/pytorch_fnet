import pdb
from util.data.dataset import DataSet
import numpy as np

class NucMaskDataSet(DataSet):
    # override
    def __getitem__(self, index):
        """Sets target to be a custom mask that is a combination of nuclear and cell masks.

        Combination mask labels:
        0 - nucleus
        1 - cell and not nucleus
        2 - not cell
        """
        vol_dna, vol_nuc_seg, vol_cell_seg = super().__getitem__(index)
        vol_classes = np.ones(vol_dna.shape, dtype=vol_dna.dtype)*2
        vol_classes[np.where(vol_cell_seg > 0)] = 1
        vol_classes[np.where(vol_nuc_seg > 0)] = 0
        return (vol_dna, vol_classes)
    
    
        
