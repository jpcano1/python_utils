from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode="zeros",
        bias=True,
        bn=True,
        activation=None,
        *args,
        **kwargs,
    ):
        """
        Initializer method
        :param in_channels: The number of in channels
        :param out_channels: The number of out channels
        :param kernel_size: The kernel size of the convolution
        :param stride: The stride of the convolution
        :param padding: The padding size of the convolution
        :param padding_mode: The mode the padding is executed
        :param bias: The bias neuron
        :param bn: Boolean to add Batch Normalization layer
        :param activation: The activation layer
        :param args: Function Arguments
        :param kwargs: Function keyword arguments
        """
        super(ConvBlock, self).__init__()

        layers = []

        # Conv Layer
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding_mode=padding_mode,
            padding=padding,
        )
        layers.append(conv_layer)

        # Batch Normalization Layer
        if bn:
            if kwargs.get("bn_params"):
                assert isinstance(kwargs.get("bn_params"), dict)
                bn_layer = nn.BatchNorm2d(out_channels, **kwargs.get("bn_params"))
            else:
                bn_layer = nn.BatchNorm2d(out_channels)
            layers.append(bn_layer)

        # Activation Layer
        if activation:
            if kwargs.get("activation_params"):
                assert isinstance(kwargs.get("activation_params"), dict)
                activation = activation(**kwargs.get("activation_params"))
            else:
                activation = activation()
            layers.append(activation)

        self.conv_block = nn.Sequential(*layers)
        del layers

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be forwarded
        :return: The tensor forwarded to the convolutional block
        """
        return self.conv_block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2, mode="bilinear", *args, **kwargs):
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
            self.upsample_layer = nn.Upsample(
                scale_factor=scale_factor, mode=mode, align_corners=True
            )
        else:
            self.upsample_layer = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        """
        The forward method
        :param x: The tensor to be
        :return: The tensor forwarded to the
        upsample block
        """
        return self.upsample_layer(x)
