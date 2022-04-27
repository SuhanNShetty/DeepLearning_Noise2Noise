import torch.empty as empty
import torch.nn.functional.fold as fold
import torch.nn.functional.unfold as unfold

torch.set_grad_enabled(False)

if __name__ == "__main__" :
    kernel_size = (2 , 2)
    x = torch.randn((1 , 3 , 32 , 32) )
    y = torch.randn((1 , 3 , 32 , 32) )
    a = torch.randn((1 ,) )

    out_channels = 4

    conv = torch.nn.Conv2d( in_channels = x.shape[1], out_channels = out_channels, kernel_size = kernel_size, bias = False )

    torch.testing.assert_allclose(a*conv(x) , conv(a*x))
    torch.testing.assert_allclose(conv(x + y) , conv(x) + conv(y))