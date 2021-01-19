from torch import nn
from .general_layers import ConvBlock
from .layers import UpBlock, DownBlock

class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, 
                 depth, output_activation=nn.Sigmoid, *args, **kwargs):
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

        # Depth must be greater than one
        assert depth > 1, f"Depth must be greater than one"
        
        # Down and up blocks
        down_blocks = []
        up_blocks = []
        
        # Keyword arguments
        pool_size = kwargs.get("pool_size", 2)
        pool_stride = kwargs.get("pool_stride", 2)
        
        # Initial Layer
        self.init_layer = ConvBlock(in_channels, init_filters, 
                                    *args, **kwargs)

        # Begin the loop
        current_filters = init_filters

        # The first down block
        down_block = DownBlock(current_filters, current_filters * 2,
                               *args, **kwargs)
        down_blocks.append(down_block)
        current_filters *= 2

        # Loop down through the depth
        for _ in range(depth - 1):
            # Create the pooling layer
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            down_blocks.append(layer)

            # Create the down block
            down_block = DownBlock(current_filters, 
                                   current_filters * 2, *args,
                                   **kwargs)
            down_blocks.append(down_block)
            current_filters *= 2

        # Loop through the up depth
        for _ in range(depth - 1):
            # Create the up block
            up_block = UpBlock(current_filters,
                               current_filters // 2, *args, **kwargs)
            up_blocks.append(up_block)
            current_filters //= 2

        # Create the module list
        self.down_blocks = nn.Sequential(*down_blocks)
        self.up_blocks = nn.Sequential(*up_blocks)

        # Final layer
        self.final_layer = ConvBlock(
            current_filters, out_channels, padding=0, 
            activation=output_activation, kernel_size=1,
            bn=True
        )
        
    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to
        the convolutional block
        """
        x = self.init_layer(x)
        x = self.down_blocks(x)
        x = self.up_blocks(x)
        return self.final_layer(x)