# Acceptable imports for project
from torch import empty , cat , arange
from torch.nn.functional import fold , unfold
from math import ceil
import math

############################################################################
class Sequential(object):
    '''
        A sequence of layers
    '''
    def __init__(self, *args):
        self.transforms = args

    def zero_grad(self):
        '''
            Set gradients to zero
        '''
        for tfm in self.transforms:
            tfm.zero_grad()
            # print('grad_weight',tfm.grad_weight)
            # print('weight',tfm.weight)

    def forward(self, x):
        for tfm in self.transforms:
            x = tfm.forward(x)
            # print('In forward', tfm.name)
        return x

    def backward(self, grad_out):
        '''
            grad_out: gradient w.r.t output 
            Collect the gradient by backpropogation 
        '''
        for tfm in self.transforms[::-1]:
            grad_out = tfm.backward(grad_out)
            # print('In backward', tfm.name) 

        return grad_out # gradient w.r.t input
    

############################################################################
class SGD(object):
    '''
        Updates the learning parameters
    '''
    def __init__(self, sqn_block, lr=1e-1, use_momentum=False, momentum=0.):
        
        self.lr = lr
        self.sqn_block = sqn_block
        # momentum params
        self.use_momentum = use_momentum
        self.momentum = 0.0
        if use_momentum and momentum < 0 :
            self.momentum = momentum;
            assert ((momentum<1) and (momentum>0)), "momentum should be between 0 and 1"
        else:
            self.momentum = 0.0
        self.set_velocity()
    
    def set_momentum(self, momentum):
        assert ((momentum<1) and (momentum>0)), "momentum should be between 0 and 1"
        self.use_momentum = True
        self.momentum = momentum
        self.set_velocity()

    def set_velocity(self):
        '''
            Initialize velocity for momentum
        '''
        self.velocity_weight = []
        self.velocity_bias = []
        for tfm in self.sqn_block.transforms:
            self.velocity_weight.append(0.*tfm.grad_weight)
            if tfm.use_bias:
                self.velocity_bias.append(0.*tfm.grad_bias)
            else:
                self.velocity_bias.append(None)
                
    def zero_grad(self):
        self.sqn_block.zero_grad()

    def step(self):
        '''
            Take one gradient step 
        '''
        for i,tfm in enumerate(self.sqn_block.transforms):
            self.velocity_weight[i] = self.momentum*self.velocity_weight[i] + self.lr*tfm.grad_weight # weight
            # update weight and bias:
            self.sqn_block.transforms[i].weight -= self.velocity_weight[i] # weight update
            if tfm.use_bias:
                self.velocity_bias[i] = self.momentum*self.velocity_bias[i] + self.lr*tfm.grad_bias  # bias
                self.sqn_block.transforms[i].bias -= self.velocity_bias[i] # bias update
            
############################################################################
class MSE(object):
    def __init__(self):
        pass 

    def forward(self, input, target):
        self.input = input
        self.target = target
        return (self.input - self.target).pow(2).mean() 

    def backward(self):
        self.grad_in = 2*(self.input-self.target)/(self.input.size(-3)*self.input.size(-2)*self.input.size(-1))
        return self.grad_in

############################################################################

class ReLU(object) :
    def __init__(self):
        self.name = "ReLU"
        self.params = ()
        self.weight = empty(1) # just for placeholding
        self.bias = empty(1)
        self.use_bias = False
        self.grad_weight = empty(1) # just for placeholding
        self.grad_bias = empty(1)

    def zero_grad(self):
        pass 

    def forward(self, input) :
        self.input = input
        self.positif_mask = (input > 0)
        return self.positif_mask*(input)

    def backward(self, gradwrtoutput) :
        self.grad_in = self.positif_mask.int()*gradwrtoutput
        return self.grad_in
    def param(self):
        return self.params

############################################################################

class Sigmoid(object) :
    def __init__(self):
        self.name = "Sigmoid"
        self.params = ()
        self.weight = empty(1) # just for placeholding
        self.bias = empty(1)
        self.use_bias = False
        self.grad_weight = empty(1) # just for placeholding
        self.grad_bias = empty(1)

    def zero_grad(self):
        pass 

    def forward(self, input) :
        self.input = input
        self.output = 1/(1 + math.e**(-input))
        return  self.output
    def backward(self, gradwrtoutput ) :
        self.grad_in = self.output * (1-self.output) * gradwrtoutput
        return self.grad_in
    def param(self) :
        return self.params

############################################################################

class Conv2d(object):
    def __init__(self, in_ch, out_ch, kernel_size = (3,3), padding = 0, stride = 1, use_bias = False, device = 'cpu'):
        self.name = "Conv2d"
        self.device =device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.k = self.kernel_size[0]
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        bound = 1/((self.k**2*self.in_ch)**0.5)
        self.weight = empty(out_ch, in_ch, self.k, self.k).uniform_(-bound, bound).to(self.device)
        self.bias = empty(out_ch).uniform_(-bound, bound).to(self.device) if use_bias else empty(out_ch).fill_(0,1).to(self.device)
        
#         self.weight = self.kernel
        self.grad_weight = 0*self.weight
        self.grad_bias = 0*self.bias
        
    def zero_grad(self):
        self.grad_weight.fill_(0.)
        if self.use_bias:
            self.grad_bias.fill_(0.)
            
    def forward(self, x):   
        
        self.batch_size = x.size(0)
        self.s_in = x.size(-1)
        self.s_out = int(math.ceil((x.size(-2)-self.k+1+self.padding*2)/(self.stride)))
        
        X_unf = unfold(x, kernel_size=(self.k, self.k), padding = self.padding, stride = self.stride)
        self.X_unf = X_unf
    
        K_expand = self.weight.view(self.out_ch, -1)
        O_expand = K_expand @ X_unf

        O = O_expand.view(self.batch_size, self.out_ch, self.s_out, self.s_out)
        self.output = O + self.bias.view(1, -1, 1, 1) if self.use_bias else O
        return self.output 
    
    def backward(self, gradwrtoutput):
        dL_dO = gradwrtoutput                                       # (B x OUT_CH x SO x SO)
        dO_dX = self.weight                                         # (OUT_CH x IN_CH x SI x SI)

        dL_dO_exp = dL_dO.reshape(self.batch_size, self.out_ch, -1) # (B x OUT_CH x (SO x SO))
        dO_dX_exp = dO_dX.reshape(self.out_ch,-1).transpose(0,1)    # (OUT_CH x (IN_CH x SI x SI))
        dL_dO_unf = dO_dX_exp @ dL_dO_exp                           # (B x (IN_CH x SI x SI) x (SO x SO))

        self.grad_in = fold(dL_dO_unf, kernel_size = (self.k, self.k), padding = self.padding, stride = self.stride, output_size = (self.s_in, self.s_in))
        
        # backward wrt weights
        dL_dO_exp = dL_dO.transpose(0,1).reshape(self.out_ch, -1) # (OUT_CH x (B x SO x SO))
        dO_dF_exp = self.X_unf.transpose(-1, -2).reshape(self.batch_size*self.s_out*self.s_out, -1) # ((B x SO x SO) x (IN_CH x K x K))
        dL_dF_exp = dL_dO_exp @ dO_dF_exp # (OUT_CH x  (IN_CH x K x K))
        
        self.grad_weight = dL_dF_exp.view(self.out_ch, self.in_ch, self.k, self.k)
        
        # backward wrt bias
        if self.use_bias:
            dO_dB_exp = (1+0*empty(self.batch_size * (self.s_out) * (self.s_out)).normal_()).to(self.device)
            self.grad_bias = dL_dO_exp @ dO_dB_exp
        else:
            self.grad_bias = None
        
        return self.grad_in

        
    def param(self) :
        return ((self.weight, self.grad_weight), (self.bias, self.grad_bias))


############################################################################

class ConvTranspose2d(object):
    def __init__(self, in_ch, out_ch, kernel_size = (3,3), padding = 0, stride = 1, use_bias = False, device='cpu', output_padding = 0):
        self.name = "ConvTranspose2d"
        self.device = device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.k_1 = self.kernel_size[0]
        self.k_2 = self.kernel_size[1]
        self.k = self.kernel_size[1]
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        bound = 1/((self.k**2*self.in_ch)**0.5)
        self.weight = empty(in_ch, out_ch, self.k_1, self.k_2).uniform_(-bound, bound).to(self.device)
        self.bias = empty(out_ch).uniform_(-bound, bound).to(self.device) if use_bias else empty(out_ch).fill_(0).to(self.device)
        self.output_padding = output_padding
        
#         self.weight = self.kernel
        self.grad_weight = 0*self.weight
        self.grad_bias = 0*self.bias     
        
    def zero_grad(self):
        self.grad_weight.fill_(0.)
        if self.use_bias:
            self.grad_bias.fill_(0.)

    def forward(self, x):
        self.x = x
        self.batch_size = x.size(0)
        self.s1 = self.x.size(-2)
        self.s2 = self.x.size(-1)
        o1 = (self.s1 - 1)*self.stride + 1 + self.k_1 - 1 - self.padding *2 + self.output_padding
        o2 = (self.s2 - 1)*self.stride + 1 + self.k_2 - 1 - self.padding *2 + self.output_padding
        
        self.o1 = o1
        self.o2 = o2
        
        x_exp = x.reshape(self.batch_size, self.in_ch, -1)
        K_exp = self.weight.reshape(self.in_ch,-1).transpose(0,1)
        out = fold(K_exp @ x_exp, kernel_size = (self.k_1, self.k_2), padding = self.padding, stride = self.stride, output_size = (o1,o2))
        
        return out + self.bias.view(1, -1, 1, 1) if self.use_bias else out
    
    def backward(self, gradwrtoutput):
        dL_dO = gradwrtoutput      # B x OUT_CH x SO x SO
        dO_dX = self.weight
        
        dL_dO_unf = unfold(dL_dO, kernel_size = (self.k_1, self.k_2), padding = self.padding, stride = self.stride) # B x (OUT_CH x K x K) x SI x SI
        dL_dX_exp = dO_dX.view(self.in_ch, -1) @ dL_dO_unf
        self.grad_in = dL_dX_exp.view(self.batch_size, self.in_ch, self.s1, self.s2)
        
        self.dL_dO_unf_K = dL_dO_unf.transpose(0,1).reshape(self.out_ch * self.k_1 * self.k_2, -1).transpose(0,1)
                                                                    # (B x SI x SI) x (OUT_CH x K x K)
        self.dO_dF_exp = self.x.transpose(0,1).reshape(self.in_ch, -1)   # IN_CH x (B x SI x SI)
        self.dL_dF_exp = self.dO_dF_exp @ self.dL_dO_unf_K                         # IN_CH x (OUT_CH x K x K)                                                                       
        self.grad_weight = self.dL_dF_exp.view(self.in_ch, self.out_ch, self.k_1, self.k_2)  # OUT_CH x IN_CH x K x K
        
        if self.use_bias:
            dL_dO_exp = dL_dO.transpose(0,1).reshape(self.out_ch, -1)
            dO_dB_exp = (1.0+0*empty(self.batch_size * (self.o1) * (self.o2)).normal_()).to(self.device)
            self.grad_bias = dL_dO_exp @ dO_dB_exp
        else:
            self.grad_bias = None
        
        return self.grad_in
        
    def param(self) :
        return ((self.weight, self.grad_weight), (self.bias, self.grad_bias))

