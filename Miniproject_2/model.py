
class convolution(object):
    def __init__(self, inp_channels, out_channels, kernel_size = (3,3), stride = (1,1), padding = (0,0), dilation = (1,1)):
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.kernel = torch.empty((out_channels, kernel_size[0], kernel_size[1])).normal_()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
    def forward(self, input):
        self.batch_size = input.size(0)
        self.input = input
        up = (input.size(-2) + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
        down = self.stride[0]
        h = torch.tensor(up/down + 1).floor().int()
        up = (input.size(-1) + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
        down = self.stride[1]
        w = torch.tensor(up/down + 1).floor().int()
        unf = unfold(input, kernel_size=self.kernel_size)
        conv = self.kernel.view(self.out_channels, -1) @ unf
#         print(conv.view(1, 1, 2, 2))
#         print(h, w)
        return conv.view(self.batch_size, self.out_channels, h, w)

    def backward(self, gradwrtoutput):
        flattened = gradwrtoutput.view(self.batch_size,self.out_channels,-1)
        print(flattened)
        pass
    def param(self) :
        return [self.kernel]