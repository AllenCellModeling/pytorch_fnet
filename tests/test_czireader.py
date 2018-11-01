from fnet.data.czireader import CziReader
import numpy.testing as npt


def test_czireader():
    path = 'data/3500000427_100X_20170120_F05_P27.czi'
    czi = CziReader(path)
    dim_to_scale = czi.get_scales()
    zyx_scales = [dim_to_scale[dim] for dim in 'zyx']
    npt.assert_almost_equal(zyx_scales, [0.29, 0.10833, 0.10833], decimal=3)
    ar_chan_0 = czi.get_volume(0)
    assert ar_chan_0.shape == (39, 512, 512)
