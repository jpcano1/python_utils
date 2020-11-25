# 1611 lines

from torch import nn
from .layers import ConvBlock, UpBlock, Encoder, Decoder

"""
Autoencoder Creator
"""
class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, depth, 
                 *args, **kwargs):
        """
        The autoencoder method for generic creation
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param init_filters: The initial filters
        :param depth: The depth of the autoencoder
        :param args: Function arguments
        :param kwargs: Function Keyword arguments
        """
        super(Autoencoder, self).__init__()
        jump = kwargs.get("jump") or 1
        
        # The encoder part
        self.encoder = Encoder(in_channels, init_filters, depth,
                               *args, **kwargs)
        # The decoder part
        self.decoder = Decoder(init_filters * 2**(jump * (depth-1)), 
                               out_channels, depth, *args, **kwargs)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to the convolutional block
        """
        # Forward to the encoder
        x = self.encoder(x)
        # Forward to the decoder
        return self.decoder(x)

"""
U-Net Creator
"""
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, depth, 
                 *args, **kwargs):
        """
        The unet method for generic creation
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param init_filters: The initial filters
        :param depth: The depth of the autoencoder
        :param args: Function arguments
        :param kwargs: Function Keyword arguments
        """
        super(UNet, self).__init__()

        # Depth must be greater than one
        assert depth > 1, f"{depth} must be greater than one"

        # Down and up layers
        self.down_layers = []
        self.up_layers = []

        # Keyword arguments
        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2
        self.jump = kwargs.get("jump") or 1

        # Initial Layer
        init_layer = ConvBlock(in_channels, init_filters, 
                               *args, **kwargs)
        self.down_layers.append(init_layer)

        current_filters = init_filters

        # Add convolutional layers while jump is greater than 1
        for j in range(self.jump - 1):
            if j == self.jump - 2:
                layer = ConvBlock(current_filters, current_filters * 2, 
                                  *args, **kwargs)
                current_filters *= 2
            else:
                layer = ConvBlock(current_filters, current_filters, 
                                  *args, **kwargs)

            self.down_layers.append(layer)

        # Loop through the down layers
        for _ in range(depth - 1):
            # Add max pool of last down block
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            self.down_layers.append(layer)

            # Double current filters if jump greater than 1
            if self.jump > 1:
                layer = ConvBlock(current_filters, current_filters, 
                                  *args, **kwargs)
            else:
                layer = ConvBlock(current_filters, current_filters * 2, 
                                  *args, **kwargs)

            # Append to the down layers
            self.down_layers.append(layer)

            # Add convolutional layers while jump greater than 1
            for j in range(self.jump - 1):
                if j == self.jump - 2:
                    # If index equals to the last layer
                    # Double filters
                    layer = ConvBlock(current_filters, current_filters * 2, 
                                      *args, **kwargs)
                else:
                    layer = ConvBlock(current_filters, current_filters, 
                                      *args, **kwargs)
                # Append to the down layers
                self.down_layers.append(layer)

            current_filters *= 2
        # Loop through the up layers
        for _ in range(depth - 1):
            # Create the transpose pool layer of last convolutional layer
            layer = UpBlock(current_filters + current_filters // 2,
                            current_filters // 2, *args, **kwargs)
            self.up_layers.append(layer)
            current_filters //= 2
             # Append layers while jump greater than 1
            for _ in range(self.jump - 1):
                layer = ConvBlock(current_filters, current_filters, 
                                  *args, **kwargs)
                self.up_layers.append(layer)

        # Final layer with the output filters
        self.final_layer = ConvBlock(current_filters, out_channels, padding=0,
                                     activation=nn.Sigmoid(), kernel_size=1, bn=False)

        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to the convolutional block
        """
        # The down convolutional blocks
        down_conv_layers = []

        # The down convolutional pass
        for idx, layer in enumerate(self.down_layers):
            x = layer(x)
            if isinstance(layer, ConvBlock):
                down_conv_layers.append(x)

        # The up convolutional pass
        for idx, layer in enumerate(self.up_layers):
            # Concatenate with the respective down layer
            if isinstance(layer, UpBlock):
                idx_layer = -(idx + self.jump + 1)
                x = layer(down_conv_layers[idx_layer], x)
            else:
                x = layer(x)
        # Forward through the last layer
        return self.final_layer(x)