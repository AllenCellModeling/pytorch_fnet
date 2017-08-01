import torch
import pdb

# changed from v5: removing padding from convolution layers
# Input and output sizes are no longer identical.
# Input should be padded with mirroring to generate larger initial chunks.

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mult_chan = 32
        depth = 3
        self.net_recurse = _Net_recurse(n_in_channels=1, mult_chan=mult_chan, depth=depth)
        self.conv_out = torch.nn.Conv3d(mult_chan, 1, kernel_size=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_rec = self.net_recurse(x)
        x_pre_out = self.conv_out(x_rec)
        x_out = self.relu(x_pre_out)
        return x_out

class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0):
        """Class for recursive definition of U-network.

        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        if depth == 0:
            self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels, k_size=1)
        else:
            self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels)
            self.conv_down = torch.nn.Conv3d(n_out_channels, n_out_channels, 2, stride=2)
            self.bn0 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu0 = torch.nn.ReLU()
            
            self.convt = torch.nn.ConvTranspose3d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
            
    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            if self.depth > 1:
                dim_reduce = (0, 0, 16, 48)[self.depth]  # TODO: hardcoded for now
                shape_cropped = (None, None, ) + tuple(dim - dim_reduce for dim in x_2conv_more.size()[2:])
                x_2conv_more_cropped = _crop_tensor(x_2conv_more, shape_cropped)
                x_cat = torch.cat((x_2conv_more_cropped, x_relu1), 1)  # concatenate
            else:
                x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out, k_size=3):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(n_in,  n_out, kernel_size=k_size)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=k_size)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_bn1 = self.bn1(x_conv1)
        x_relu1 = self.relu1(x_bn1)
        x_conv2 = self.conv2(x_relu1)
        x_bn2 = self.bn2(x_conv2)
        x_relu2 = self.relu2(x_bn2)
        return x_relu2

def _crop_tensor(t, dims_crop):
    """Return Tensor formed from cropping t to dims_crop.

    Cropping is taken from the center of each dimension.

    t - tensor
    dims_crop - tuple with length equal to number of dims in t.
    """
    dims_t = t.size()
    result = t
    for i in range(len(dims_crop)):
        if dims_crop[i] is None:
            continue
        start = dims_t[i]//2 - dims_crop[i]//2
        result = result.narrow(i, start, dims_crop[i])
    return result
