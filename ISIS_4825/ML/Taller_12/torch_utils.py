from torch import nn
import torch

"""
Autoencoder Creator
"""
def create_convolution_block(in_channels, out_channels, kernel_size=3, 
                            stride=1, padding=1, padding_mode="zeros"):
    x = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, stride=stride, 
                padding_mode=padding_mode, padding=padding)
    return x

def get_upsample(scale_factor=2, mode="bilinear"):
    if mode != "nearest":
        x = nn.Upsample(scale_factor=scale_factor, mode=mode, 
                    align_corners=True)
    else:
        x = nn.Upsample(scale_factor=scale_factor, mode=mode)
    return x

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        kernel_size = kwargs.get("kernel_size") or 3
        stride = kwargs.get("stride") or 1
        padding = kwargs.get("padding") or 1
        padding_mode = kwargs.get("padding_mode") or "zeros"
        activation = kwargs.get("activation") or nn.LeakyReLU(0.2)
        bn = kwargs.get("bn") or 0

        layers = []
        conv2d_layer = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding_mode=padding_mode, 
                                padding=padding)
        layers.append(conv2d_layer)
        if torch.rand() < bn:
            bn_layer = nn.BatchNorm2d(out_channels)
            layers.append(bn_layer)

        layers.append(activation)
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)
        
class Encoder(nn.Module):
    def __init__(self, in_channels, init_filters, depth, **kwargs):
        super(Encoder, self).__init__()
        layers = []
        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2

        jump = kwargs.get("jump") or 1

        init_layer = create_convolution_block(in_channels, init_filters)
        layers.append(init_layer)

        layer = kwargs.get("activation") or nn.LeakyReLU(0.2)
        layers.append(layer)

        current_filters = init_filters
        
        for _ in range(depth - 1):
            # Pooling Layer

            for _ in range(jump - 1):
                # Convolution Block
                layer = create_convolution_block(current_filters,
                                                 current_filters * 2)
                layers.append(layer)

                # Activation Block
                layer = nn.LeakyReLU(0.2) or kwargs.get("activation")
                layers.append(layer)

                current_filters *= 2

            # Pooling Block
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            layers.append(layer)

            # Convolution Block
            layer = create_convolution_block(current_filters, 
                                             current_filters * 2)
            layers.append(layer)

            # Activation Block
            layer = nn.LeakyReLU(0.2) or kwargs.get("activation")
            layers.append(layer)

            current_filters *= 2

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, init_filters, out_channels, depth, **kwargs):
        super(Decoder, self).__init__()
        layers = []

        scale_factor = kwargs.get("scale_factor") or 2 
        mode = kwargs.get("mode") or "bilinear"

        jump = kwargs.get("jump") or 1

        current_filters = init_filters

        for _ in range(depth - 1):    
            for _ in range(jump - 1):
                layer = create_convolution_block(current_filters,
                                                 current_filters // 2)
                layers.append(layer)

                layer = kwargs.get("activation") or nn.LeakyReLU(0.2)
                layers.append(layer)

                current_filters //= 2

            layer = get_upsample(scale_factor=scale_factor, mode=mode)
            layers.append(layer)
            layer = create_convolution_block(current_filters,
                                             current_filters // 2)
            layers.append(layer)
            layer = kwargs.get("activation") or nn.LeakyReLU(0.2)
            layers.append(layer)

            current_filters //= 2

        layer = create_convolution_block(current_filters, 
                                         out_channels)
        layers.append(layer)
        layer = nn.Sigmoid()
        layers.append(layer)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, depth, 
                 *args, **kwargs):
        super(Autoencoder, self).__init__()
        jump = kwargs.get("jump") or 1

        self.encoder = Encoder(in_channels, init_filters, depth,
                               *args, **kwargs)
        self.decoder = Decoder(init_filters * 2**(jump * (depth-1)), 
                               out_channels, depth, *args, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)