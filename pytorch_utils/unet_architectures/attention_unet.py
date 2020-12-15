from torch import nn

from .general_layers import ConvBlock
from .layers import AttentionUpBlock, DownBlock

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters,
                 depth, output_activation=nn.Sigmoid, *args, **kwargs):
        """

        :param in_channels:
        :param out_channels:
        :param init_filters:
        :param depth:
        :param output_activation:
        :param args:
        :param kwargs:
        """
        super(AttentionUNet, self).__init__()

        assert depth > 1

        down_blocks = []
        up_blocks = []

        # Keyword arguments
        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2

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

            # Create the down block
            down_block = DownBlock(current_filters, 
                                   current_filters * 2, *args,
                                   **kwargs)
            down_blocks.append(down_block)
            current_filters *= 2

        for _ in range(depth - 1):
            up_block = AttentionUpBlock(current_filters, current_filters // 2, 
                                        current_filters // 2, *args, **kwargs)
            up_blocks.append(up_block)
            current_filters //= 2

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final_layer = ConvBlock(
            current_filters, out_channels, padding=0, 
            activation=output_activation, kernel_size=1
        )

    def forward(self, x):
        """

        :param x:
        :return:
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