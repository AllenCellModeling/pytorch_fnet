import fnet.nn_modules.fnet_nn_3d_params
import pdb

class Net(fnet.nn_modules.fnet_nn_3d_params.Net):
    def __init__(self):
        super().__init__(depth=4, mult_chan=32)

