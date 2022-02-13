import torch
import torch.nn as nn

from model.conv_norm_relu_block import ConvolutionInstanceNormLeakyRelu


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        #nn.ReflectionPad2d(1)
        self.rb_first = ConvolutionInstanceNormLeakyRelu(in_channels=dim, 
                                                         out_channels=dim,
                                                         kernel_size=3, 
                                                         stride=1,
                                                         padding=1, #0, 
                                                         is_norm=True, 
                                                         relu_type='rely')
        
        self.rb_second = ConvolutionInstanceNormLeakyRelu(in_channels=dim, 
                                                          out_channels=dim,
                                                          kernel_size=3, 
                                                          stride=1,
                                                          padding=1, #0, 
                                                          is_norm=True, 
                                                          relu_type='none')
    
    def forward(self, input):
        return input + self.rb_second(self.rb_first(input)) 



class Generator(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3):
        super(Generator, self).__init__()
        self.c7s1_64 = ConvolutionInstanceNormLeakyRelu(in_channels=num_in_ch, 
                                                        out_channels=64,
                                                        kernel_size=7, 
                                                        stride=1,
                                                        padding=3, #0, 
                                                        is_norm=True, 
                                                        relu_type='relu')
        
        self.d128 = ConvolutionInstanceNormLeakyRelu(in_channels=64, 
                                                     out_channels=128,
                                                     kernel_size=3, 
                                                     stride=2,
                                                     padding=1, 
                                                     is_norm=True, 
                                                     relu_type='relu')

        self.d256 = ConvolutionInstanceNormLeakyRelu(in_channels=128, 
                                                     out_channels=256,
                                                     kernel_size=3, 
                                                     stride=2,
                                                     padding=1, 
                                                     is_norm=True, 
                                                     relu_type='relu') 
        
        self.R256x9 = nn.Sequential(*[ResidualBlock(dim=256) for i in range(9)])

        self.u128 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128, 
                               kernel_size=3, 
                               stride=2, 
                               padding=1, 
                               output_padding=1),
            nn.InstanceNorm2d(128, eps=1e-05),
            nn.ReLU(inplace=True)
            )

        self.u64 = nn.Sequential(
             nn.ConvTranspose2d(in_channels=128,
                                out_channels=64, 
                                kernel_size=3, 
                                stride=2, 
                                padding=1, 
                                output_padding=1),
             nn.InstanceNorm2d(64, eps=1e-05),
             nn.ReLU(inplace=True)
             )        
        
        self.c7s1_3 = ConvolutionInstanceNormLeakyRelu(in_channels=64, 
                                                       out_channels=num_out_ch,
                                                       kernel_size=7, 
                                                       stride=1,
                                                       padding=3, #0, 
                                                       is_norm=False, 
                                                       relu_type='none')
        
    def forward(self,input):
        output = self.c7s1_64(input)
        output = self.d128(output)
        output = self.d256(output)
        output = self.R256x9(output)
        output = self.u128(output)
        output = self.u64(output)
        output = self.c7s1_3(output)
        return torch.tanh(output)
