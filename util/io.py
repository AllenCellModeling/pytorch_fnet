import os
from aicsimage.io import omeTifReader
import numpy as np
import pdb

def read_tifs(path_dir, filename_bits):
    """Read TIFs in folder and return as tuple of numpy arrays.

    Parameters:
    path_dir - path to directory of TIFs
    filename_bits - tuple/list of strings that should match the ends of the target filenames
                       e.g., (_trans.tif, _dna.tif)

    Returns:
    tuple of numpy arrays
    """
    assert isinstance(filename_bits, (tuple, list))
    print('reading TIFs from', path_dir)
    file_list = [i.path for i in os.scandir(path_dir) if i.is_file()]  # order is arbitrary

    paths_to_read = []
    for bit in filename_bits:
        matches = [f for f in file_list if (bit in f)]
        if len(matches) != 1:
            print('WARNING: incorrect number of files found for pattern {} in {}'.format(bit, path_dir))
            return None
        paths_to_read.append(matches[0])
    
    vol_list = []
    for path in paths_to_read:
        fin = omeTifReader.OmeTifReader(path)
        try:
            fin = omeTifReader.OmeTifReader(path)
        except:
            print('WARNING: could not read file:', path)
            return None
        vol_list.append(fin.load().astype(np.float32)[0, ])  # Extract the sole channel
    if len(vol_list) != len(filename_bits):
        print('WARNING: did not read in correct number of files')
        return None
    return tuple(vol_list)
