from torch import nn

"""
Autoencoder Creator
"""
def create_convolution_block(in_channels, out_channels, kernel_size=3, 
                            stride=1, padding=1, padding_mode="zeros"):
    x = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, stride=stride, 
                padding_mode=padding_mode, padding=padding)
    return x

def get_upsample(scale_factor=2, mode="bilinear"):
    if mode != "nearest":
        x = nn.Upsample(scale_factor=scale_factor, mode=mode, 
                    align_corners=True)
    else:
        x = nn.Upsample(scale_factor=scale_factor, mode=mode)
    return x

class Encoder(nn.Module):
    def __init__(self, in_channels, init_filters, depth, **kwargs):
        super(Encoder, self).__init__()
        layers = []
        pool_size = kwargs.get("pool_size") or 2
        pool_stride = kwargs.get("pool_stride") or 2

        init_layer = create_convolution_block(in_channels, init_filters)
        layers.append(init_layer)

        layer = kwargs.get("activation") or nn.LeakyReLU(0.2)
        layers.append(layer)
        
        layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        layers.append(layer)

        i = 0
        while i < depth-2:
            layer = create_convolution_block(init_filters*2**i, 
                                             init_filters*2**(i+1))
            layers.append(layer)
            layer = nn.LeakyReLU(0.2) or kwargs.get("activation")
            layers.append(layer)
            layer = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
            layers.append(layer)
            i += 1
        layer = create_convolution_block(init_filters*2**i, 
                                        init_filters*2**(i+1))
        layers.append(layer)

        layer = kwargs.get("activation") or nn.LeakyReLU(0.2)
        layers.append(layer)
        
        if kwargs.get("batch_normalization"):
            layer = nn.BatchNorm2d(init_filters*2**(i+1))
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, init_filters, out_channels, depth, **kwargs):
        super(Decoder, self).__init__()
        layers = []

        i = 0
        while i < depth - 1:
            layer = get_upsample()
            layers.append(layer)
            layer = create_convolution_block(init_filters // 2**i, 
                                                  init_filters // 2**(i+1))
            layers.append(layer)
            layer = nn.LeakyReLU(0.2) or kwargs.get("activation")
            layers.append(layer)
            i += 1

        layer = create_convolution_block(init_filters // 2**i, 
                                              out_channels)
        layers.append(layer)
        layer = nn.Sigmoid()
        layers.append(layer)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters, depth, 
                 *args, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, init_filters, depth,
                               *args, **kwargs)
        self.decoder = Decoder(init_filters * 2**(depth-1), out_channels,
                               depth, *args, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)