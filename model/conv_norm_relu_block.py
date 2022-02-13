import torch
import torch.nn as nn


class ConvolutionInstanceNormLeakyRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 is_norm=True, relu_type='leaky_rely'):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(in_channels=in_channels, 
                                 out_channels=out_channels, 
                                 kernel_size=kernel_size, 
                                 stride=stride, 
                                 padding=padding, 
                                 padding_mode="reflect"))
        if is_norm:
            modules.append(nn.InstanceNorm2d(out_channels, eps=1e-05))

        if relu_type == 'leaky_rely':
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif relu_type == 'rely':            
            modules.append(nn.ReLU(inplace=True))
        elif relu_type == 'none':
            modules.append(nn.Identity())     

        self.conv_unit = nn.Sequential(*modules)    

    def forward(self, input):
        return self.conv_unit(input)

