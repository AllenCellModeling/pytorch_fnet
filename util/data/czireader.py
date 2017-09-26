import aicsimage.io as io
import os

class CziReader(object):
    def __init__(self, path_czi):
        super().__init__()
        # Currently, expect to deal only with CZI files where 'B' and '0' dimensions are size 1
        self.czi_reader = io.cziReader.CziReader(path_czi)
        self.czi_np = self.czi_reader.czi.asarray()
        self._check_czi()
        self.axes = ''.join(map(chr, self.czi_reader.czi.axes))
        path_basename = os.path.basename(path_czi)
        # print('{} | axes: {} | shape: {} | dtype: {}'.format(
        #     path_basename,
        #     self.axes,
        #     self.czi_np.shape,
        #     self.czi_np.dtype)
        # )
        self.metadata = self.czi_reader.get_metadata()

    def get_size(self, dim_sel):
        dim = -1
        if isinstance(dim_sel, int):
            dim = dim_sel
        elif isinstance(dim_sel, str):
            dim = self.axes.find(dim_sel)
        assert dim >= 0
        return self.czi_np.shape[dim]

    def _check_czi(self):
        # all dims not corresponding to TCZYX should be zero (maybe?)
        for i in range(len(self.czi_reader.czi.axes)):
            dim_label = self.czi_reader.czi.axes[i]
            if dim_label not in b'TCZYX':
                assert self.czi_np.shape[i] == 1

    def get_volume(self, chan, time_slice=None):
        """Returns the image volume for the specified channel."""
        slices = []
        for i in range(len(self.czi_reader.czi.axes)):
            dim_label = self.czi_reader.czi.axes[i]
            if dim_label in b'C':
                slices.append(chan)
            elif dim_label in b'T':
                slices.append(0)
            elif dim_label in b'ZYX':
                slices.append(slice(None))
            else:
                slices.append(0)
        return self.czi_np[slices]
    
