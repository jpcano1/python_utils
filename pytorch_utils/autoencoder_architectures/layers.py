from torch import nn

from .general_layers import ConvBlock, UpsampleBlock


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param out_channels: The number of out channels.
        :param args: Function arguments.
        :param kwargs: Function keyword arguments.
        """
        super(DownBlock, self).__init__()

        # Create the layers list
        layers = []

        # The number of convolutions per block
        jump = kwargs.get("jump", 2)

        # The initial layer of the jump loop
        init_layer = ConvBlock(in_channels, out_channels, *args, **kwargs)

        layers.append(init_layer)

        # Append convolutional blocks if
        # jump is greater than 1
        for _ in range(jump - 1):
            layer = ConvBlock(out_channels, out_channels, *args, **kwargs)
            layers.append(layer)

        self.down_block = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to down block
        """
        return self.down_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(UpBlock, self).__init__()

        jump = kwargs.get("jump", 2)
        layers = []

        # The upsampling layer
        self.upsample = UpsampleBlock(*args, **kwargs)

        # The first convolutional layer
        layer = ConvBlock(in_channels, out_channels, *args, **kwargs)
        layers.append(layer)

        # Append convolutional blocks
        # if jump is greater than 1
        for _ in range(jump - 1):
            layer = ConvBlock(out_channels, out_channels, *args, **kwargs)
            layers.append(layer)
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward method
        :param down_block: The tensor from the down block
        :param last_block: The tensor from the last block
        :return:
        """
        x = self.upsample(x)
        return self.conv_block(x)
