# Acceptable imports for project
from torch import empty , cat , arange, ceil
from torch.nn.functional import fold , unfold
import torch

############################################################################
class SGD(object):
	'''
		Updates the learning parameters
		Disclaimer: We used https://github.com/pytorch/pytorch/blob/cd9b27231b51633e76e28b6a34002ab83b0660fc/torch/optim/sgd.py#L63
		as reference for writing our version
	'''
	def __init__(self, sqn_block, lr=1e-6, use_momentum=False, damping=0.):
		self.lr = lr
		# momentum params
		self.damping = damping
		self.sqn_block = sqn_block
		if use_momentum is True:
			self.set_velocity()
			if damping < 0 :
				raise ValueError("Damping can not be < 0 for SGD with momentum")	

	def set_velocity(self):
		'''
			Initialize velocity for momentum
		'''
		self.velocity = []
		for tfm in self.sqn_block.transforms:
			self.velocity.append((0*tfm.params.grad_weight,0*tfm.params.grad_bias))

    def zero_grad(self):
        self.sqn_block.zero_grad()


	def step(self):
		'''
			Take one gradient step 
			To Do: Check if the original tfm are updated
		'''
		for i,tfm in enumerate(self.sqn_block.transforms):
			# set momentum
			self.velocity[i][0] = self.damping*self.velocity[i][0] + tfm.params[i].grad_weight # weight
			self.velocity[i][1] = self.damping*self.velocity[i][1] + tfm.params[i].grad_bias  # bias
			# update weight and bias:
			tfm.params.weight -= self.lr*self.velocity[i][0] # weight
			tfm.params.bias -= self.lr*self.velocity[i][1] # bias
			# # zero the grad:
			# tfm.params.grad_weight.fill_(0.) #weight
			# tfm.params.grad_bias.fill_(0.) # bias

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

class Sequential(object):
	'''
		A sequence of layers
	'''
	def __init__(self, *args):
		self.transforms = args
		# To do: consider skip connection

    def zero_grad(self):
        '''
            Set gradients to zero
        '''
        for tfm in self.transforms:
            tfm.params.grad_weight.fill_(0.)
            tfm.params.grad_bias.fill_(0)


	def forward(self, x):
		for tfm in self.transforms:
			x = tfm.forward(x)
		return x

	def backward(self, grad_out):
		'''
			grad_out: gradient w.r.t output 
			Collect the gradient by backpropogation 
		'''
		for tfm in self.transforms[::-1]:
			grad_out = tfm.backward(grad_out)	
		return grad_out # gradient w.r.t input

############################################################################

class ReLU(object) :
    def __init__(self):
        self.params = ()
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
		self.params = ()
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
    def __init__(self, in_ch, out_ch, kernel_size = (3,3), padding = 0, stride = 1, use_bias = False, device='cpu'):
        self.device =device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.k = self.kernel_size[0]
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.kernel = empty(out_ch, in_ch, self.k, self.k).normal_().to(self.device)
        self.bias = empty(out_ch).normal_() if use_bias else torch.zeros(out_ch).to(self.device)

        self.params.weight = self.kernel
        self.params.bias = self.bias
        
    def forward(self, x): 
    	# Update the kernel and bias
    	self.kernel = self.params.weight  
        self.bias = self.params.bias

        self.batch_size = x.size(0)
        self.s_in = x.size(-1)
        self.s_out = (ceil((x.size(-2)-self.k+1+self.padding*2)/(self.stride))).long()
        
        X_unf = unfold(x, kernel_size=(self.k, self.k), padding = self.padding, stride = self.stride)
        
        self.x = x
        self.X_unf = X_unf
    
        K_expand = self.kernel.view(self.out_ch, -1)
        O_expand = K_expand @ X_unf
        
        O = O_expand.view(self.batch_size, self.out_ch, self.s_out, self.s_out)
        return O + self.bias.view(1, -1, 1, 1) if self.use_bias else O
    
    def backward(self, gradwrtoutput):
        dL_dO = gradwrtoutput                                       # (B x OUT_CH x SO x SO)
        dO_dX = self.kernel                                         # (OUT_CH x IN_CH x SI x SI)

        dL_dO_exp = dL_dO.reshape(self.batch_size, self.out_ch, -1) # (B x OUT_CH x (SO x SO))
        dO_dX_exp = dO_dX.reshape(self.out_ch,-1).transpose(0,1)    # (OUT_CH x (IN_CH x SI x SI))
        dL_dO_unf = dO_dX_exp @ dL_dO_exp                           # (B x (IN_CH x SI x SI) x (SO x SO))

        dL_dX = fold(dL_dO_unf, kernel_size = (self.k, self.k), padding = self.padding, stride = self.stride, output_size = (self.s_in, self.s_in))
        
        # backward wrt weights
        dL_dO_exp = dL_dO.transpose(0,1).reshape(self.out_ch, -1) # (OUT_CH x (B x SO x SO))
        dO_dF_exp = self.X_unf.transpose(-1, -2).reshape(self.batch_size*self.s_out*self.s_out, -1) # ((B x SO x SO) x (IN_CH x K x K))
        dL_dF_exp = dL_dO_exp @ dO_dF_exp # (OUT_CH x  (IN_CH x K x K))
        
        self.dL_dF = dL_dF_exp.view(self.out_ch, self.in_ch, self.k, self.k)
        
        # backward wrt bias
        if self.use_bias:
            dO_dB_exp = 1+0*empty(self.batch_size * (self.s_out) * (self.s_out)).to(self.device)
            self.dL_dB = dL_dO_exp @ dO_dB_exp
        else:
            self.dL_dB = None

        # save the params
        self.grad_in = dL_dX
        self.params.grad_weight = dL_dF
        self.params.grad_bias = dL_dB
        
        return dL_dX
        
    def param(self) :
        return ((self.params.weight,self.params.grad_weight),(self.params.bias, elf.params.grad_bias))


############################################################################

class TransposeConv2d(object):
    def __init__(self, in_ch, out_ch, kernel_size = (3,3), padding = 0, stride = 1, use_bias = False, device='cpu'):
        self.device = device
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.k_1 = self.kernel_size[0]
        self.k_2 = self.kernel_size[1]
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.kernel = empty(in_ch, out_ch, self.k_1, self.k_2).normal_().to(self.device)
        self.bias = empty(out_ch).normal_() if use_bias else 0*empty(out_ch).to(self.device)

    def forward(self, x):
    	# Update the kernel and bias
    	self.kernel = self.params.weight  
        self.bias = self.params.bias

        self.x = x
        self.batch_size = x.size(0)
        self.s1 = self.x.size(-2)
        self.s2 = self.x.size(-1)
        o1 = (self.s1 - 1)*self.stride + 1 + self.k_1 - 1 - self.padding *2
        o2 = (self.s2 - 1)*self.stride + 1 + self.k_2 - 1 - self.padding *2
        
        self.o1 = o1
        self.o2 = o2
        
        x_exp = x.reshape(self.batch_size, self.in_ch, -1)
        K_exp = self.kernel.reshape(self.in_ch,-1).transpose(0,1)
        O_unf = K_exp @ x_exp
        out = fold(O_unf, kernel_size = (self.k_1, self.k_2), padding = self.padding, stride = self.stride, output_size = (o1,o2))
        
        return out + self.bias.view(1, -1, 1, 1) if self.use_bias else out
    
    def backward(self, gradwrtoutput):
        dL_dO = gradwrtoutput      # B x OUT_CH x SO x SO
        dO_dX = self.kernel
        
        dL_dO_unf = unfold(dL_dO, kernel_size = (self.k_1, self.k_2), padding = self.padding, stride = self.stride)
                                   # B x (OUT_CH x K x K) x SI x SI
        dO_dX_exp = dO_dX.view(self.in_ch, -1)
        dL_dX_exp = dO_dX_exp @ dL_dO_unf
        self.dL_dX = dL_dX_exp.view(self.batch_size, self.in_ch, self.s1, self.s2)
        
        self.dL_dO_unf_K = dL_dO_unf.transpose(0,1).reshape(self.out_ch * self.k_1 * self.k_2, -1).transpose(0,1)
                                                                    # (B x SI x SI) x (OUT_CH x K x K)
        self.dO_dF_exp = self.x.transpose(0,1).reshape(self.in_ch, -1)   # IN_CH x (B x SI x SI)
        self.dL_dF_exp = self.dO_dF_exp @ self.dL_dO_unf_K                         # IN_CH x (OUT_CH x K x K)                                                                       
        self.dL_dF = self.dL_dF_exp.view(self.in_ch, self.out_ch, self.k_1, self.k_2)  # OUT_CH x IN_CH x K x K
        
        dL_dO_exp = dL_dO.transpose(0,1).reshape(self.out_ch, -1)
        dO_dB_exp = 1+0*empty(self.batch_size * (self.o1) * (self.o2)).to(self.device)
        self.dL_dB = dL_dO_exp @ dO_dB_exp

        # save the params
        self.grad_in = dL_dX
        self.params.grad_weight = dL_dF
        self.params.grad_bias = dL_dB
        
        return self.dL_dX

    def param(self) :
        return ((self.params.weight,self.params.grad_weight),(self.params.bias, self.params.grad_bias))

