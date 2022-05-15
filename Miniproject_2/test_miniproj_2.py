"""
Test of the different modules
"""

import torch
from model import Conv
if __name__ == '__main__':
    kernel_size=(2, 2)
    
    x = torch.randn((1, 3, 32, 32))
    y = torch.randn((1, 3, 32, 32))
    a = torch.randn((1,))
    
    out_channels = 4
    
    conv_pytorch = torch.nn.Conv2d(in_channels = x.shape[1], out_channels = out_channels, kernel_size = kernel_size, bias = False)
    
    conv = Conv()
    
    torch.testing.assert_allclose(a * conv_pytorch(x), conv(a * x ))
    
    
