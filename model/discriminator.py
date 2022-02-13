#Implementation of the Discriminator for the CycleGAN model.

import torch
import torch.nn as nn

from model.conv_norm_relu_block import ConvolutionInstanceNormLeakyRelu


class Discriminator(nn.Module):
    def __init__(self, num_in_channels=3):
        super(Discriminator, self).__init__()
        self.c64 = ConvolutionInstanceNormLeakyRelu(num_in_channels, 
                                                    out_channels=64,
                                                    kernel_size=4, 
                                                    stride=2,
                                                    padding=1, 
                                                    is_norm=False, 
                                                    relu_type='leaky_rely')
        
        self.c128 = ConvolutionInstanceNormLeakyRelu(in_channels=64, 
                                                     out_channels=128,
                                                     kernel_size=4, 
                                                     stride=2,
                                                     padding=1)
        
        self.c256 = ConvolutionInstanceNormLeakyRelu(in_channels=128, 
                                                     out_channels=256,
                                                     kernel_size=4, 
                                                     stride=2,
                                                     padding=1)
        
        self.c512 = ConvolutionInstanceNormLeakyRelu(in_channels=256, 
                                                     out_channels=512,
                                                     kernel_size=4,  
                                                     stride=1,
                                                     padding=1)
        
        self.layer_last = ConvolutionInstanceNormLeakyRelu(in_channels=512, 
                                                           out_channels=1,
                                                           kernel_size=4, 
                                                           stride=1,
                                                           padding=1,
                                                           is_norm=False, 
                                                           relu_type='none')
        
    def forward(self,input):
        output = self.c64(input)
        output = self.c128(output)
        output = self.c256(output)
        output = self.c512(output)
        output = self.layer_last(output)
        return torch.sigmoid(output)
