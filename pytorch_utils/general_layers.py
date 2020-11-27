from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param out_channels: The number of out channels
        :param padding: The padding size
        :param args: Function arguments
        :param kwargs: Function Keyword arguments
        """
        super(ConvBlock, self).__init__()
        # The kernel size
        kernel_size = kwargs.get("kernel_size") or 3
        # The stride size
        stride = kwargs.get("stride") or 1

        # Padding Mode
        padding_mode = kwargs.get("padding_mode") or "zeros"
        # Activation Function
        activation = kwargs.get("activation") or nn.LeakyReLU(0.2)
        # Batch Normalization
        bn = kwargs.get("bn") or False

        layers = []

        # Convolutional layer creation
        conv2d_layer = nn.Conv2d(in_channels=in_channels, 
                                out_channels=out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding_mode=padding_mode, 
                                padding=padding)
        layers.append(conv2d_layer)
        if bn:
            bn_layer = nn.BatchNorm2d(out_channels)
            layers.append(bn_layer)

        layers.append(activation)
        # The creation of the layers from a Sequential module.
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to the convolutional block
        """
        return self.conv_block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2, mode="bilinear", 
                 *args, **kwargs):
        """
        Initializer method
        :param scale_factor: The factor of upsampling
        :param mode: The mode of upsampling,
        generally an interpolation method
        :param args: Function arguments
        :param kwargs: Function keyword arguments
        """
        super(UpsampleBlock, self).__init__()

        # Conditional modes
        if mode != "nearest":
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor, 
                                              mode=mode, align_corners=True)
        else:
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor,
                                              mode=mode)
            
    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be
        :return: The tensor forwarded to the
        upsample block
        """
        return self.upsample_layer(x)