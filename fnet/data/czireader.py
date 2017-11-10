import aicsimage.io as io
import os
import pdb

def get_czi_metadata(element, tag_list):
    """
    element - (xml.etree.ElementTree.Element)
    tag_list - list of strings
    """
    if len(tag_list) == 0:
        return None
    if len(tag_list) == 1:
        if tag_list[0] == 'attrib':
            return [element.attrib]
        if tag_list[0] == 'text':
            return [element.text]
    values = []
    for sub_ele in element:
        if sub_ele.tag == tag_list[0]:
            if len(tag_list) == 1:
                values.extend([sub_ele])
            else:
                retval = get_czi_metadata(sub_ele, tag_list[1:])
                if retval is not None:
                    values.extend(retval)
    if len(values) == 0:
        return None
    return values


class CziReader(object):
    def __init__(self, path_czi):
        with io.cziReader.CziReader(path_czi) as reader:
            # self.czi_reader = io.cziReader.CziReader(path_czi)
            self.czi_np = reader.czi.asarray()
            self.axes = ''.join(map(chr, reader.czi.axes))
            self.metadata = reader.get_metadata()

    def get_size(self, dim_sel):
        dim = -1
        if isinstance(dim_sel, int):
            dim = dim_sel
        elif isinstance(dim_sel, str):
            dim = self.axes.find(dim_sel)
        assert dim >= 0
        return self.czi_np.shape[dim]

    def get_scales(self):
        tag_list = 'Metadata.Scaling.Items.Distance'.split('.')
        dict_scales = {}
        for entry in get_czi_metadata(self.metadata, tag_list):
            dim = entry.attrib.get('Id')
            if (dim is not None) and (dim.lower() in 'zyx'):
                scale = None
                try:
                    # convert from m/px to um/px
                    scale = float(get_czi_metadata(entry, ['Value'])[0].text)*10**6
                except:
                    pass
                dict_scales[dim.lower()] = scale
        return dict_scales

    def _check_czi(self):
        pass

    def get_volume(self, chan, time_slice=None):
        """Returns the image volume for the specified channel."""
        slices = []
        for i in range(len(self.axes)):
            dim_label = self.axes[i]
            if dim_label in 'C':
                slices.append(chan)
            elif dim_label in 'T':
                if time_slice is None:
                    slices.append(0)
                else:
                    slices.append(time_slice)
            elif dim_label in 'ZYX':
                slices.append(slice(None))
            else:
                slices.append(0)
        return self.czi_np[slices]
    
