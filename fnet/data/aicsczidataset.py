from fnet.data.czireader import CziReader
from fnet.data.fnetdataset import FnetDataset
import importlib
import numpy as np
import os
import pandas as pd
import pdb
import tifffile
import torch

def _to_objects(slist):
    olist = list()
    for s in slist:
        if not isinstance(s, str):
            olist.append(s)
            continue
        s_split = s.split('.')
        for idx_part, part in enumerate(s_split):
            if not part.isidentifier():
                break
        importee = '.'.join(s_split[:idx_part])
        so = '.'.join(s_split[idx_part:])
        if len(importee) > 0:
            module = importlib.import_module(importee)
            so = 'module.' + so
        olist.append(eval(so))
    return olist

class AICSCziDataset(FnetDataset):
    """Dataset for AICS CZI files."""

    def __init__(self,
                 dataframe: pd.DataFrame = None,
                 path_csv: str = None,
                 content_signal:str = None,
                 content_target:str = None,
                 transform_signal:int = None,
                 transform_target:int = None,
                 path_cache_dir: str = None,
    ):
        """
        Parameters:
            dataframe: Pandas DataFrame with column "path_czi". Each row is a dataset entry.
            path_csv: Path to a csv to be turned into a Pandas DataFrame. This argument will be overridden by "dataframe".
            content_signal: Semantic content of signal channel.
                If specified, the channel number for both the signal and target will be determined automatically.
                Allowed values: [None, 'bright-field', 'bf']
            content_target: Semantic content of target channel.
                Allowed values: [None, 'cmdrp', 'egfp', 'dna', 'h3342']
            transform_signal: transformation or list of transformations to be applied to signal image.
            transform_target: transformation or list of transformations to be applied to target image.
        """
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
            
        self.content_signal = content_signal.lower() if content_signal is not None else content_signal
        self.content_target = content_target.lower() if content_target is not None else content_target
        self.transform_signal = transform_signal
        self.transform_target = transform_target
        self.path_cache_dir = path_cache_dir
        self._find_channels = False

        assert 'path_czi' in self.df.columns
        if self.content_signal is None:
            assert all(i in self.df.columns for i in ['channel_signal', 'channel_target'])

        if self.path_cache_dir is not None:
            if not os.path.exists(self.path_cache_dir):
                os.makedirs(self.path_cache_dir)
        if self.content_signal is not None:
            self._find_channels = True

        # If transform_signa/transform_target is a string, convert to objects
        self.transform_signal = _to_objects(self.transform_signal)
        self.transform_target = _to_objects(self.transform_target)
        print('DEBUG: transform_signal', self.transform_signal)
        print('DEBUG: transform_target', self.transform_target)

    def _get_path_cached(self, path_czi, channel, transforms):
        basename = os.path.basename(path_czi)
        tlist = 'none'
        if transforms is not None:
            tlist = '_'.join([repr(t).replace(' ', '') for t in transforms])
        path_cached = os.path.join(
            self.path_cache_dir,
            '{:s}_chan{:d}_{:s}.tiff'.format(basename, channel, tlist),
        )
        return path_cached

    def __getitem__(self, idx):
        if self._find_channels:
            raise NotImplementedError

        index_val = self.df.index[idx]
        
        path_czi = self.df.loc[index_val, 'path_czi']
        channel_signal = self.df.loc[index_val, 'channel_signal']
        channel_target = self.df.loc[index_val, 'channel_target']
        flip_y = self.df.loc[index_val, :].get('flip_y', -1) > 0
        flip_x = self.df.loc[index_val, :].get('flip_x', -1) > 0

        czi = None
        data = list()
        for channel, transform in ((channel_signal, self.transform_signal), (channel_target, self.transform_target)):
            element = None
            if np.isnan(channel):
                continue
            if self.path_cache_dir is not None:
                path_cached = self._get_path_cached(path_czi, channel, transform)
                if os.path.exists(path_cached):
                    print('DEBUG: used cached file:', path_cached)
                    element = tifffile.imread(path_cached)
            if element is None:
                if czi is None:
                    czi = CziReader(path_czi)
                element = czi.get_volume(channel)
                if transform is not None:
                    for t in transform:
                        element = t(element)
                        print('DEBUG: After transform', t, element.shape)
                element = element[np.newaxis, ]  # Add "channel" dimension to all images
                if self.path_cache_dir is not None:
                    tifffile.imsave(path_cached, element)
                    print('saved:', path_cached)
                    
            # Optional augmentations
            if flip_y:
                print('flipping y')
                element = np.flip(element, axis=-2)
            if flip_x:
                print('flipping x')
                element = np.flip(element, axis=-1)
                
            data.append(element)
        
        data = [torch.tensor(ar.astype(np.float), dtype=torch.float32) for ar in data]
        return data
    
    def __len__(self):
        return len(self.df)

    def get_information(self, idx: int) -> dict:
        return self.df.iloc[idx, :].to_dict()
