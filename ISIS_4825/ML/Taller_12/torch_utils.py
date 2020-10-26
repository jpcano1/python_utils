from torch import nn
import numpy as np
import torch

"""
Autoencoder Creator
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, *args, **kwargs):
        super(ConvBlock, self).__init__()
        kernel_size = kwargs.get("kernel_size") or 3
        stride = kwargs.get("stride") or 1

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
        if np.random.rand() < bn:
            bn_layer = nn.BatchNorm2d(out_channels)
            layers.append(bn_layer)

        layers.append(activation)
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2, mode="bilinear", *args, **kwargs):
        super(UpsampleBlock, self).__init__()
        if mode != "nearest":
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor, 
                                              mode=mode, align_corners=True)
        else:
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor,
                                              mode=mode)
            
    def forward(self, x):
        return self.upsample_layer(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(UpBlock, self).__init__()
        
        activation = kwargs.get("activation") or nn.LeakyReLU(0.2)
        self.upsample = UpsampleBlock()
        self.conv_layer = ConvBlock(in_channels, out_channels, 
                                    activation=activation)

    def forward(self, down_conv, last_conv):
        last_conv = self.upsample(last_conv)
        x = torch.cat((last_conv, down_conv), dim=1)
        return self.conv_layer(x)
        
class Encoder(nn.Module):
    def __init__(self, in_channels, init_filters, depth, **kwargs):
        super(Encoder, self).__init__()
        layers = []
        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2

        bn = kwargs.get("bn") or 0

        jump = kwargs.get("jump") or 1

        init_layer = ConvBlock(in_channels, init_filters)
        layers.append(init_layer)

        current_filters = init_filters
        
        for _ in range(depth - 1):
            # Pooling Layer

            for _ in range(jump - 1):
                # Convolution Block
                layer = ConvBlock(current_filters,
                                  current_filters * 2, bn=bn)
                layers.append(layer)

                current_filters *= 2

            # Pooling Block
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            layers.append(layer)

            # Convolution Block
            layer = ConvBlock(current_filters,
                              current_filters * 2, bn=bn)
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

        bn = kwargs.get("bn") or 0

        jump = kwargs.get("jump") or 1

        current_filters = init_filters

        for _ in range(depth - 1):    
            for _ in range(jump - 1):
                layer = ConvBlock(current_filters,
                                  current_filters // 2, bn=bn)
                layers.append(layer)

                current_filters //= 2

            layer = UpsampleBlock(scale_factor=scale_factor, mode=mode)
            layers.append(layer)
            layer = ConvBlock(current_filters,
                              current_filters // 2)
            layers.append(layer)

            current_filters //= 2

        layer = ConvBlock(current_filters, out_channels, 
                          activation=nn.Sigmoid(), bn=bn)
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

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, depth, 
                 *args, **kwargs):
        super(UNet, self).__init__()


        assert depth > 1, f"{depth} must be greater than one"
        self.down_layers = []
        self.up_layers = []

        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2

        init_layer = ConvBlock(in_channels, init_filters)
        self.down_layers.append(init_layer)

        current_filters = init_filters

        for _ in range(depth - 1):
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            self.down_layers.append(layer)

            layer = ConvBlock(current_filters,current_filters*2)
            self.down_layers.append(layer)
            current_filters *= 2

        for _ in range(depth - 1):
            layer = UpBlock(current_filters + current_filters // 2,
                            current_filters // 2)
            self.up_layers.append(layer)
            current_filters //= 2

        self.final_layer = ConvBlock(current_filters, out_channels, padding=0,
                                     activation=nn.Sigmoid(), kernel_size=1)

        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

    def forward(self, x):
        down_conv_layers = []

        for layer in self.down_layers:
            x = layer(x)
            if isinstance(layer, ConvBlock):
                down_conv_layers.append(x)
        for idx, layer in enumerate(self.up_layers):
            x = layer(down_conv_layers[-(idx+2)], x)

        return self.final_layer(x)