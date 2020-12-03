from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, padding_mode="zeros", bias=False,
                 *args, **kwargs):
        """
        Initializer method
        :param in_channels: The number of in channels.
        :param out_channels: The number of out channels
        :param padding: The padding size
        :param args: Function arguments
        :param kwargs: Function Keyword arguments
        """
        super(ConvBlock, self).__init__()
        # Activation Function

        if kwargs.get("activation"):
            activation = kwargs.get("activation")
        else:
            activation = nn.LeakyReLU(0.2,
                                      inplace=True)

        self.conv_block = nn.Sequential(
            # Convolutional Layer
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding_mode=padding_mode, 
                      padding=padding, bias=bias),
            # Batch Normalization Layer
            nn.BatchNorm2d(out_channels),
            # Activation Layer
            activation
        )

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