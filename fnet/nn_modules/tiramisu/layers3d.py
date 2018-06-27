import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate = 0.2):
        super().__init__()
        
        self.main = nn.Sequential(nn.BatchNorm3d(in_channels),
                                    nn.ReLU(True),
                                    nn.Conv3d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True)
                                 )
        
        self.last = None
        if dropout_rate > 0:
            self.last = nn.Dropout3d(dropout_rate, inplace=True)

    def forward(self, x):
        x = checkpoint(self.main, x)
        
        if self.last is not None:
            x = self.last(x)
        
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False, dropout_rate = 0.2):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate, dropout_rate = dropout_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, dropout_rate = 0.2):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, in_channels,
                                          kernel_size=2, stride=2,
                                          padding=0, bias=True))
        
        self.drop = None
        if dropout_rate > 0:
            self.drop = nn.Dropout3d(dropout_rate, inplace=True)
        
        #removed max pooling and replaced with convolution with stride 2
        # self.add_module('maxpool', nn.MaxPool3d(2))

    def forward(self, x):
        x = checkpoint(super().forward, x)
        
        if self.drop is not None:
            x = self.drop(x)
            
        return x 


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = checkpoint(self.convTrans, x)
        out = center_crop(out, skip.size(2), skip.size(3), skip.size(4))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers, dropout_rate = 0.2):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(
            in_channels, growth_rate, n_layers, upsample=True, dropout_rate = dropout_rate))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width, max_depth):
    _, _, h, w, d = layer.size()
    xyz1 = (w - max_width) // 2
    xyz2 = (h - max_height) // 2
    xyz3 = (d - max_depth) // 2
    return layer[:, :, xyz2:(xyz2 + max_height), xyz1:(xyz1 + max_width), xyz3:(xyz3 + max_depth)]
