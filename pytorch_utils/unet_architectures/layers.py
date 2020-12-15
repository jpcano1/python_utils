from .general_layers import ConvBlock, UpsampleBlock

import torch
from torch import nn
from torch.nn import functional as F

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
        jump = kwargs.get("jump") or 2

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

        jump = kwargs.get("jump") or 2
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
        result = self.conv_block(x)
        for _ in range(self.t):
            result = self.conv_block(x + result)
        return result

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
        jump = kwargs.get("jump") or 2

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

        jump = kwargs.get("jump") or 2
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
    def __init__(self, in_channels, out_channels,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(RRUpBlock, self).__init__()

        jump = kwargs.get("jump") or 2
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

"""
Attention Blocks
"""
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, 
                 inter_channels, sub_sample_factor=2, 
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param gating_channels: The number of channels
        from the gating signal
        :param inter_channels: The number of inter channels
        :param sub_sample_factor: The factor of subsampling
        theta convolution
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(AttentionBlock, self).__init__()

        sub_sample_kernel = sub_sample_factor

        # Theta convolution for down convolution
        self.theta = ConvBlock(
            in_channels=in_channels, out_channels=inter_channels, 
            kernel_size=sub_sample_kernel, stride=sub_sample_factor, 
            padding=0, bias=False, bn=False
        )

        # The phi convolution for the gating signal
        self.phi = ConvBlock(
            in_channels=gating_channels, out_channels=inter_channels, 
            kernel_size=1, stride=1, padding=0, bn=False
        )

        # The psi convolution for the conjuction of the
        # last convolutions
        self.psi = ConvBlock(
            in_channels=inter_channels, out_channels=1, kernel_size=1, 
            stride=1, padding=0, bn=False
        )

        # Final transformation for the psi
        # and x conjunction
        self.out_transform = ConvBlock(
            in_channels=in_channels, out_channels=in_channels, 
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, g):
        """
        The forward method
        :param x: The x tensor from the down convolution
        :param g: The tensor from the gating signal
        :return: The attention transformed tensor
        """
        x_shape = x.shape

        # Theta convolution
        theta_x = self.theta(x)
        # Phi convolution
        phi_g = self.phi(g)
        f = F.relu(theta_x + phi_g, inplace=True)

        # The psi convolution
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # Interpolation for conjunction
        sigm_psi_f = F.interpolate(sigm_psi_f, size=x_shape[2:], 
                                   mode="bilinear", align_corners=True)
        # Conjunction between psi tensor and down conv tensor
        y = sigm_psi_f.expand_as(x) * x
        return self.out_transform(y)

class AttentionUpBlock(nn.Module):
    def __init__(self, last_channels, down_channels, out_channels,
                 *args, **kwargs):
        """
        Initializer method
        :param last_channels: The number of channels from the last convolution
        :param down_channels: The number of channels from the down convolution
        :param out_channels: The out channels from the up block
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(AttentionUpBlock, self).__init__()

        # The number of convolutions before max pooling
        jump = kwargs.get("jump") or 2

         # List of layers
        layers = []
        # Gating convolution
        self.gating = ConvBlock(
            in_channels=last_channels, 
            out_channels=last_channels // 2, *args, **kwargs
        )

        # Upsampling layer
        self.upsample = UpsampleBlock(*args, **kwargs)

        # Attention block
        self.attention = AttentionBlock(
            in_channels=last_channels // 2, 
            gating_channels=down_channels, 
            inter_channels=last_channels // 2, 
            *args, **kwargs
        )

        # Convolutional layer
        layer = ConvBlock(last_channels + down_channels, 
                          out_channels, *args, **kwargs)
        layers.append(layer)

        for _ in range(jump - 1):
            layer = ConvBlock(out_channels, out_channels, *args, **kwargs)
            layers.append(layer)
        
        self.conv_block = nn.Sequential(*layers)

    def forward(self, down_block, last_block):
        """
        Forward method
        :param down_block: The tensor from the down block
        :param last_block: The tensor from the last block
        :return: the tensor forwarded
        """
        # The gating tensor
        gating_x = self.gating(last_block)
        # Attention tensor
        attention_x = self.attention(down_block, gating_x)
        # Upsample tensor
        last_block = self.upsample(last_block)
        x = torch.cat((last_block, attention_x), dim=1)
        return self.conv_block(x)