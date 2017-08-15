import aicsimage.io as io

class CziReader(object):
    def __init__(self, file_path):
        super().__init__()
        # Currently, expect to deal only with CZI files where 'B' and '0' dimensions are size 1
        self.czi_reader = io.cziReader.CziReader(file_path)
        self.czi_np = self.czi_reader.czi.asarray()
        self._check_czi()
        self.axes = ''.join(map(chr, self.czi_reader.czi.axes))
        print('CZI axes: {} | shape: {}'.format(self.axes, self.czi_np.shape))

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

    def get_volume(self, chan):
        """Returns the image volume for the specified channel."""
        if self.czi_reader.hasTimeDimension:
            raise NotImplementedError  # TODO: handle case of CZI images with T dimension
        slices = []
        for i in range(len(self.czi_reader.czi.axes)):
            dim_label = self.czi_reader.czi.axes[i]
            if dim_label in b'C':
                slices.append(chan)
            elif dim_label in b'ZYX':
                slices.append(slice(None))
            else:
                slices.append(0)
        return self.czi_np[slices]
