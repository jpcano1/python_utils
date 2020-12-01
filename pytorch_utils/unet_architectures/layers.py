from .general_layers import ConvBlock, UpsampleBlock

import torch
from torch import nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 *args, **kwargs):
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
        jump = kwargs.get("jump") or 1

        # The initial layer of the jump loop
        init_layer = ConvBlock(in_channels, out_channels, 
                               *args, **kwargs)
        
        layers.append(init_layer)

        # Append convolutional blocks if
        # jump is greater than 1
        for _ in range(jump - 1):
            layer = ConvBlock(out_channels, out_channels, 
                              *args, **kwargs)
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
    def __init__(self, in_channels, out_channels,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(UpBlock, self).__init__()

        jump = kwargs.get("jump") or 1
        layers = []

        # The upsampling layer
        self.upsample = UpsampleBlock(*args, **kwargs)

        # The first convolutional layer
        layer = ConvBlock(in_channels, out_channels, 
                          *args, **kwargs)
        layers.append(layer)

        # Append convolutional blocks
        # if jump is greater than 1
        for _ in range(jump - 1):
            layer = ConvBlock(out_channels, out_channels, 
                              *args, **kwargs)
            layers.append(layer)
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, down_block, last_block):
        """
        The forward method
        :param down_block: The tensor from the down block
        :param last_block: The tensor from the last block
        :return:
        """
        last_block = self.upsample(last_block)
        x = torch.cat((last_block, down_block), dim=1)
        return self.conv_block(x)


"""
Recurrent Layers
"""
class RecurrentConvBlock(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param args: Function arguments
        :param kwargs: Function Keyword arguments
        """
        super(RecurrentConvBlock, self).__init__()
        self.t = kwargs.get("t") or 2
        self.conv_block = ConvBlock(in_channels, in_channels, 
                                    *args, **kwargs)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to the convolutional block
        """
        x = self.conv_block(x)
        for _ in range(self.t - 1):
            out = self.conv_block(x + x)
        return out

class RecurrentDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param out_channels: The number of out channels.
        :param args: Function arguments.
        :param kwargs: Function keyword arguments.
        """
        super(RecurrentDownBlock, self).__init__()

        # Create the layers list
        layers = []

        # The number of convolutions per block
        jump = kwargs.get("jump") or 1

        # The initial layer of the jump loop
        init_layer = ConvBlock(in_channels, out_channels, 
                               *args, **kwargs)
        
        layers.append(init_layer)

        # Append recurrent convolutional
        # blocks if jump is greater than 1
        for _ in range(jump - 1):
            layer = RecurrentConvBlock(out_channels, 
                                       *args, **kwargs)
            layers.append(layer)

        self.down_block = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to down block
        """
        return self.down_block(x)

class RecurrentUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(RecurrentUpBlock, self).__init__()

        jump = kwargs.get("jump") or 1
        layers = []

        # The upsampling layer
        self.upsample = UpsampleBlock(*args, **kwargs)

        # The first convolutional layer
        layer = ConvBlock(in_channels, out_channels, 
                          *args, **kwargs)
        layers.append(layer)

        # Append convolutional blocks
        # if jump is greater than 1
        for _ in range(jump - 1):
            layer = RecurrentConvBlock(out_channels, 
                                       *args, **kwargs)
            layers.append(layer)
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, down_block, last_block):
        """
        The forward method
        :param down_block: The tensor from the down block
        :param last_block: The tensor from the last block
        :return:
        """
        last_block = self.upsample(last_block)
        x = torch.cat((last_block, down_block), dim=1)
        return self.conv_block(x)

"""
Recurrent Residual Layers
"""
class RRDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, jump=2,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param out_channels: The number of out channels.
        :param jump: The number of convolutions per block
        :param args: Function arguments.
        :param kwargs: Function keyword arguments.
        """
        super(RRDownBlock, self).__init__()

        assert jump > 1, "Jump must be greater than one"
        # Create the layers list
        layers = []

        # The initial layer of the jump loop
        self.init_layer = ConvBlock(in_channels, out_channels, 
                                    *args, **kwargs)

        # Append convolutional blocks if
        # jump is greater than 1
        for _ in range(jump - 1):
            layer = RecurrentConvBlock(out_channels, 
                                       *args, **kwargs)
            layers.append(layer)

        self.recurrent_block = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to down block
        """
        x1 = self.init_layer(x)
        x2 = self.recurrent_block(x1)
        out = x1 + x2
        return out

class RRUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, jump=2,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(RRUpBlock, self).__init__()

        jump = kwargs.get("jump") or 1
        layers = []

        # The upsampling layer
        self.upsample = UpsampleBlock(*args, **kwargs)

        # The first convolutional layer
        self.init_layer = ConvBlock(in_channels, out_channels, 
                                    *args, **kwargs)

        # Append convolutional blocks
        # if jump is greater than 1
        for _ in range(jump - 1):
            layer = RecurrentConvBlock(out_channels, 
                                       *args, **kwargs)
            layers.append(layer)
        self.recurrent_block = nn.Sequential(*layers)
    
    def forward(self, down_block, last_block):
        """
        The forward method
        :param down_block: The tensor from the down block
        :param last_block: The tensor from the last block
        :return:
        """
        last_block = self.upsample(last_block)
        x = torch.cat((last_block, down_block), dim=1)
        x1 = self.init_layer(x)
        x2 = self.recurrent_block(x1)
        out = x1 + x2
        return out