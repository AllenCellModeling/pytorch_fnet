import torch.utils.data
import aicsimage.io as io
from fnet.data.fnetdataset import FnetDataset
import pandas as pd
import numpy as np

import pdb

import fnet.transforms as transforms

class CziMovieDataset(FnetDataset):
    """Dataset for CZI movie files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, 
                    transform_source = [transforms.normalize],
                    transform_target = None,
                    n_source_points = 1,
                    n_offset = 1,
                    n_target_points = 1,
                    exclude_scenes = 'None',
                    source_points = None):
                
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
            
        self.transform_source = transform_source
        self.transform_target = transform_target
        self.n_source_points = n_source_points
        self.n_offset = n_offset
        self.n_target_points = n_target_points
        
        assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])
        #assume one file with multiple scenes or multiple files with one scene each
        element = self.df.iloc[0, :]
        with io.czifile.CziFile(element['path_czi'], multifile = False) as czi:
            axes = ''.join(map(chr, czi.axes))
            if axes.find('S') >= 0:    
                self.scene_count = czi.shape[axes.find('S')]
            else:
                self.scene_count = len(self.df)
            first_block = next(czi.subblocks())
            block_dict = {str(dim.dimension)[2]: dim.size for dim in first_block.dimension_entries if str(dim.dimension)[2] in 'YX'}
            y_size = block_dict['Y']
            x_size = block_dict['X']
            t_size = czi.shape[axes.find('T')]
            z_size = czi.shape[axes.find('Z')]
            
        self.val_mode = False
        self.transform_val = [transforms.Propper()]
        self.scenes = [i for i in range(self.scene_count)]
        self.val_scenes = eval(exclude_scenes)
        if self.val_scenes is not None:
            if type(self.val_scenes) is not list:
                self.val_scenes = [self.val_scenes]
            for scene_to_exclude in self.val_scenes:
                self.scenes.remove(scene_to_exclude)
                
        self.time_slice_min = element['time_slice_min'] if 'time_slice_min' in self.df.columns else 0
        self.time_slice_max = min(element['time_slice_max'] , t_size) if 'time_slice_max' in self.df.columns else t_size
        self.scene_size = [self.time_slice_max - self.time_slice_min, z_size, y_size, x_size]
        if source_points is None:
            self.source_points = [i + n_offset for i in range(self.n_source_points)]
            self.source_points.reverse()
        else:
            source_points = [int(i) for i in source_points.split('_')]
            self.source_points = source_points

    def __getitem__(self, index):
        signal = np.zeros(self.scene_size)
        target = np.zeros(self.scene_size)
        if self.val_mode:
            scene = self.val_scenes[index]
        else:
            scene = self.scenes[index]
        if len(self.df) == 1:
            element = self.df.iloc[0, :]
        else:
            element = self.df.iloc[scene, :]
            
        channel_signal = element['channel_signal']
        channel_target = element['channel_target']
        times_to_include = eval(element['time_slices']) if 'time_slices' in self.df.columns else slice(None)
        path_czi = element['path_czi']
        with io.czifile.CziFile(path_czi, multifile = False) as czi:
            for block in czi.subblocks():          
                block_dict = {str(dim.dimension)[2]: dim.start for dim in block.dimension_entries if str(dim.dimension)[2] in 'STCZ'}
                if 'S' not in block_dict or block_dict['S'] == scene:
                    t_slice = block_dict['T']
                    if self.time_slice_max is not None and t_slice >= self.time_slice_max:
                        continue
                    if self.time_slice_min is not None:
                        t_slice -= self.time_slice_min
                        if t_slice < 0:
                            continue
                    z_slice = block_dict['Z']
                    channel = block_dict['C']
                    if channel == channel_signal:
                        signal[t_slice, z_slice, :, :] = np.squeeze(block.data())
                    if channel == channel_target:
                        target[t_slice, z_slice, :, :] = np.squeeze(block.data())
 
        signal = signal[times_to_include]
        time_slices_signal = [sl for sl in signal]
        if self.transform_source is not None:
            for t in self.transform_source:
                time_slices_signal = [t(ts) for ts in time_slices_signal]
            if self.val_mode:
                for t in self.transform_val:
                    time_slices_signal = [t(ts) for ts in time_slices_signal]

        im_out = list()
        im_out.append(np.array(time_slices_signal))

        target = target[times_to_include]
        time_slices_target = [sl for sl in target] 

        if self.transform_target is not None:
            for t in self.transform_target: 
                time_slices_target = [t(ts) for ts in time_slices_target]
            if self.val_mode:
                for t in self.transform_val:
                    time_slices_target = [t(ts) for ts in time_slices_target]
        
        target = np.array(time_slices_target)
        im_out.append(target)
        im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]
        return im_out

    def __len__(self):
        return len(self.scenes) if not self.val_mode else len(self.val_scenes)

    def get_information(self, index: int) -> dict:
        info = self.df.iloc[0, :].to_dict()
        info['scene'] = index
        return info
    
    def transform_patch(self, patch):
        assert len(patch) == 2
        n_slices = patch[0].size()[0]
        
        max_lag = self.source_points[0]
        assert n_slices > max_lag + self.n_target_points - 1
        target_start = np.random.randint(max_lag, n_slices - self.n_target_points + 1)
        target_end = target_start + self.n_target_points
        signal_slab = [target_start - lag for lag in self.source_points]
        
        signal_patch = patch[0][torch.LongTensor(signal_slab)]
        target_patch = patch[1][slice(target_start, target_end)]

        return [signal_patch, target_patch]
    
    def get_prediction_batch(self, index):
        signal, target = self[index]
        n_slices = signal.size()[0]
        max_lag = self.source_points[0]
        n_batches = n_slices - self.n_target_points + 1 - max_lag
        assert n_batches > 0
        signal_batch = [torch.unsqueeze(signal[torch.LongTensor([i - lag for lag in self.source_points])], 0) for i in range(max_lag, max_lag + n_batches)]
        target_batch = [torch.unsqueeze(target[slice(i, i + self.n_target_points)], 0) for i in range(max_lag, max_lag + n_batches)]
        return [torch.cat(signal_batch), torch.cat(target_batch)]
