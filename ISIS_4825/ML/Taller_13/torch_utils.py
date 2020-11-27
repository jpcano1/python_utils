# 1611 lines

from torch import nn
from .layers import (ConvBlock, UpBlock, 
                     Encoder, Decoder, 
                     DownBlock)

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
        assert depth > 1, f"Depth must be greater than one"
        
        # Down and up blocks
        down_blocks = []
        up_blocks = []

        # Keyword arguments
        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2
        
        # Initial Layer
        self.init_layer = ConvBlock(in_channels, init_filters, 
                                    *args, **kwargs)

        current_filters = init_filters

        down_block = DownBlock(current_filters, current_filters * 2,
                               *args, **kwargs)
        down_blocks.append(down_block)
        current_filters *= 2

        for _ in range(depth - 1):
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            down_blocks.append(layer)

            down_block = DownBlock(current_filters, 
                                   current_filters * 2, *args,
                                   **kwargs)
            down_blocks.append(down_block)
            current_filters *= 2
        
        for _ in range(depth - 1):
            up_block = UpBlock(current_filters + current_filters // 2,
                            current_filters // 2, *args, **kwargs)
            up_blocks.append(up_block)
            current_filters //= 2
        
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final_layer = ConvBlock(current_filters, out_channels, padding=0, 
                                     activation=nn.Sigmoid(), kernel_size=1,
                                     bn=True)
        
    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to the convolutional block
        """
        x = self.init_layer(x)
        
        down_blocks = []

        for idx, block in enumerate(self.down_blocks):
            x = block(x)
            if isinstance(block, DownBlock):
                down_blocks.append(x)

        for idx, block in enumerate(self.up_blocks):
            idx_block = -(idx + 2)
            x = block(down_blocks[idx_block], x)
        
        return self.final_layer(x)