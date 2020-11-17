from torch import nn
from .layers import ConvBlock, UpBlock, Encoder, Decoder

"""
Autoencoder Creator
"""
class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, depth, 
                 *args, **kwargs):
        """

        :param in_channels:
        :type in_channels:
        :param out_channels:
        :type out_channels:
        :param init_filters:
        :type init_filters:
        :param depth:
        :type depth:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        super(Autoencoder, self).__init__()
        jump = kwargs.get("jump") or 1

        self.encoder = Encoder(in_channels, init_filters, depth,
                               *args, **kwargs)
        self.decoder = Decoder(init_filters * 2**(jump * (depth-1)), 
                               out_channels, depth, *args, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

"""
U-Net Creator
"""
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, depth, 
                 *args, **kwargs):
        """

        :param in_channels:
        :type in_channels:
        :param out_channels:
        :type out_channels:
        :param init_filters:
        :type init_filters:
        :param depth:
        :type depth:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        super(UNet, self).__init__()

        assert depth > 1, f"{depth} must be greater than one"
        self.down_layers = []
        self.up_layers = []

        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2
        self.jump = kwargs.get("jump") or 1

        init_layer = ConvBlock(in_channels, init_filters)
        self.down_layers.append(init_layer)
        bn = kwargs.get("bn") or 0

        current_filters = init_filters

        for j in range(self.jump - 1):
            if j == self.jump - 2:
                layer = ConvBlock(current_filters, current_filters * 2, bn=bn)
                current_filters *= 2
            else:
                layer = ConvBlock(current_filters, current_filters)

            self.down_layers.append(layer)

        for _ in range(depth - 1):
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            self.down_layers.append(layer)

            if self.jump > 1:
                layer = ConvBlock(current_filters, current_filters, bn=bn)
            else:
                layer = ConvBlock(current_filters, current_filters * 2, bn=bn)

            self.down_layers.append(layer)

            for j in range(self.jump - 1):
                if j == self.jump - 2:
                    layer = ConvBlock(current_filters, current_filters * 2, bn=bn)
                else:
                    layer = ConvBlock(current_filters, current_filters)

                self.down_layers.append(layer)

            current_filters *= 2

        for _ in range(depth - 1):
            layer = UpBlock(current_filters + current_filters // 2,
                            current_filters // 2)
            self.up_layers.append(layer)
            current_filters //= 2

            for _ in range(self.jump - 1):
                layer = ConvBlock(current_filters, current_filters, bn=bn)
                self.up_layers.append(layer)

        self.final_layer = ConvBlock(current_filters, out_channels, padding=0,
                                     activation=nn.Sigmoid(), kernel_size=1, bn=bn)

        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

    def forward(self, x):
        """

        :param x:
        :type x:
        :return:
        :rtype:
        """
        down_conv_layers = []

        for idx, layer in enumerate(self.down_layers):
            x = layer(x)
            if isinstance(layer, ConvBlock):
                down_conv_layers.append(x)
        for idx, layer in enumerate(self.up_layers):
            if isinstance(layer, UpBlock):
                idx_layer = -(idx + self.jump + 1)
                x = layer(down_conv_layers[idx_layer], x)
            else:
                x = layer(x)
            
        return self.final_layer(x)