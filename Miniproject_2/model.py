# Acceptable imports for project
from torch import empty , cat , arange, save, load
from torch.nn.functional import fold , unfold
import math

from pathlib import Path      # For saving and loading reasons
from tqdm import tqdm         # For ease of use
        
class Model():
    def __init__(self, device = 'cpu'):
        # State the device
        self.device = device

        # State useful values for the modules
        self.batch_size = 100
        self.in_ch = 3             # Input and last output channels
        self.m = 100               # Inner layers channels
        self.k = 3                 # Kernel size
        self.lr = 1e-2             # Learning rate
        stride = 2
        padding = 1
        output_padding = 1
        use_bias=True
        
        # Modules to be used, 2 Conv2d and 2 TransposeConv2d
        self.conv1 = Conv2d(self.in_ch,self.m, kernel_size = self.k, stride=stride, padding=padding, device=self.device, use_bias = use_bias)
        self.conv2 = Conv2d(self.m, 2*self.m, kernel_size = self.k, stride=stride, padding=padding, device=self.device, use_bias = use_bias)
        self.tconv1 = TransposeConv2d(2*self.m, self.m, kernel_size = self.k, stride=stride, padding=padding, device=self.device, output_padding=output_padding, use_bias = use_bias)
        self.tconv2 = TransposeConv2d(self.m, self.in_ch, kernel_size = self.k, stride=stride, padding=padding, device=self.device, output_padding=output_padding, use_bias = use_bias)
        
        # Create the model using the Sequential
        self.model = Sequential(self.conv1,ReLU(),
            self.conv2,ReLU(),
            self.tconv1,ReLU(),
            self.tconv2,Sigmoid()
            )
        
        # Initialize the optimizer
        self.optimizer = SGD(self.model, lr = self.lr)
        self.optimizer.set_momentum(0.9)
        
        # Create the loss
        self.mse = MSE()
        
    def train(self,noisy_imgs_1, noisy_imgs_2, num_epochs = 10):
        # Retrieve images and prepare them
        noisy_imgs_1 = noisy_imgs_1.to(self.device).float()/256
        noisy_imgs_2 = noisy_imgs_2.to(self.device).float()/256
        inp_1 = noisy_imgs_1.clone().split(self.batch_size)
        tar_1 = noisy_imgs_2.clone().split(self.batch_size)
        
        # Prepare a tensor to retrieve the loss during training
        self.loss_train = empty(num_epochs).fill_(0).to(self.device)
        
        for e in tqdm(range(num_epochs)):
            for i in (range(len(inp_1))):
                # Retrieve output of the model
                out = self.model.forward(inp_1[i])
                # Compute loss
                loss = self.mse.forward(out, tar_1[i])
                # State the gradient to zero
                self.optimizer.zero_grad()
                # Compute the backward for the model
                out_grad = self.model.backward(self.mse.backward())
                # Compute a step of the SGD
                self.optimizer.step()
                # Retrieve the loss for this batch
                self.loss_train[e] += loss

    
    def predict(self,noisy_imgs):
        # Retrieve images and prepare them
        noisy_imgs = noisy_imgs.clone().to(self.device).float()/256
        
        return (self.model.forward(noisy_imgs)*256).byte()
    
    def save_dict(self, path):
        dict = {}
        for i, j in enumerate(self.model.transforms):
            dict[str(i) + '_' + j.name] = (j.weight, j.bias)
        save(dict, path)

    def load_dict(self, path, device = 'cpu'):
        
        dict = load(path)
        for i, j in enumerate(self.model.transforms):
            j.weight= dict[str(i) + '_' + j.name][0].to(device)
            j.bias = dict[str(i) + '_' + j.name][1].to(device)
        
        return self.model
    
    def save_model(self):
        f = Path('bestmodel.pth')
        p = Path('Miniproject_2')
        path = p / f if Path.is_dir(p) else f

        self.save_dict(path)
    
    def load_pretrained_model(self):
        f = Path('bestmodel.pth')
        p = Path('Miniproject_2')
        path = p / f if Path.is_dir(p) else f
        

        self.model = self.load_dict(path, device = self.device)


############################################################ Modules ##############################################################
# Sequential module
class Sequential(object):
    '''
        A sequence of modules
    '''
    def __init__(self, *args):
        self.transforms = args

    def zero_grad(self):
        '''
            Set gradients to zero for all modules
        '''
        for tfm in self.transforms:
            tfm.zero_grad()

    def forward(self, x):
        '''
            Compute the forward pass of each modules
        '''
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
# SGD module
class SGD(object):
    '''
        Updates the learning parameters
    '''
    def __init__(self, sqn_block, lr=1e-1, use_momentum=False, momentum=0.):
        
        self.lr = lr                           # Learning rate
        self.sqn_block = sqn_block             # Sequential block
        self.use_momentum = use_momentum       # Momentum usage
        self.momentum = 0.0                    # Momentum to 0 by default
        if use_momentum and momentum < 0 :
            self.momentum = momentum;          # Update momentum if needed
            assert ((momentum<1) and (momentum>0)), "momentum should be between 0 and 1"
        # Prepare velocity tensors for further use
        self.set_velocity()
    
    def set_momentum(self, momentum):
        '''
            Set the momentum if needed by user
        '''
        assert ((momentum<1) and (momentum>0)), "momentum should be between 0 and 1"
        self.use_momentum = True               # Update momentum usage
        self.momentum = momentum               # Update momentum

    def set_velocity(self):
        '''
            Initialize velocity for momentum
        '''
        # Lists to keep velocity tensors
        self.velocity_weight = []
        self.velocity_bias = []
        # Create tensors of zero, of the size of the parameters
        for tfm in self.sqn_block.transforms:
            self.velocity_weight.append(0.*tfm.grad_weight)
            if tfm.use_bias:
                self.velocity_bias.append(0.*tfm.grad_bias)
            else:
                self.velocity_bias.append(None)
                
    def zero_grad(self):
        '''
            Put gradients to zero before step
        '''
        self.sqn_block.zero_grad()

    def step(self):
        '''
            Take one gradient step for all modules
        '''
        for i,tfm in enumerate(self.sqn_block.transforms):
            # Update velocity of weights following momentum
            self.velocity_weight[i] = self.momentum*self.velocity_weight[i] + self.lr*tfm.grad_weight
            # Update weights
            self.sqn_block.transforms[i].weight -= self.velocity_weight[i]
            if tfm.use_bias:
                # Update velocity of bias following momentum
                self.velocity_bias[i] = self.momentum*self.velocity_bias[i] + self.lr*tfm.grad_bias
                # Update bias
                self.sqn_block.transforms[i].bias -= self.velocity_bias[i]
            
############################################################################
# Mean Squared Error module
class MSE(object):
    def __init__(self):
        pass 

    def forward(self, input, target):
        # Store input and target for use in backward
        self.input = input
        self.target = target
        # Return the MSE
        return (self.input - self.target).pow(2).mean() 

    def backward(self):
        # Compute gradient from forward values
        self.grad_in = 2*(self.input-self.target)/(self.input.size(-3)*self.input.size(-2)*self.input.size(-1))
        return self.grad_in

############################################################################
# ReLU module
class ReLU(object) :
    def __init__(self):
        self.name = "ReLU"
        self.params = ()
        # Weight and biases are unused, except for the step, but have no impact
        self.weight = empty(1)
        self.bias = empty(1)
        self.use_bias = False
        self.grad_weight = empty(1)
        self.grad_bias = empty(1)

    def zero_grad(self):
        pass 

    def forward(self, input) :
        self.input = input
        # Get mask on positive values of input
        self.positif_mask = (input > 0)
        # Return max(0,x)
        return self.positif_mask*(input)

    def backward(self, gradwrtoutput) :
        self.grad_in = self.positif_mask.int()*gradwrtoutput
        return self.grad_in
    
    def param(self):
        return self.params

############################################################################
# Sigmoid module
class Sigmoid(object) :
    def __init__(self):
        self.name = "Sigmoid"
        self.params = ()
        # Weight and biases are unused, except for the step, but have no impact
        self.weight = empty(1)
        self.bias = empty(1)
        self.use_bias = False
        self.grad_weight = empty(1)
        self.grad_bias = empty(1)

    def zero_grad(self):
        pass 

    def forward(self, input) :
        self.input = input
        # Compute the Sigmoid
        self.output = 1/(1 + math.e**(-input))
        return  self.output
    
    def backward(self, gradwrtoutput ) :
        # Compute the Sigmoid derivative
        self.grad_in = self.output * (1-self.output) * gradwrtoutput
        return self.grad_in
    
    def param(self) :
        return self.params

############################################################################
# Convolution module
class Conv2d(object):
    def __init__(self, in_ch, out_ch, kernel_size = 3, padding = 0, stride = 1, use_bias = False, device = 'cpu'):
        self.name = "Conv2d"
        self.device =device
        self.use_bias = use_bias
        self.in_ch = in_ch               # Input channels
        self.out_ch = out_ch             # Output channels
        self.k = kernel_size             # Kernel size
        self.stride = stride
        self.padding = padding
        # Initialization of the parameters
        bound = 1/((self.k**2*self.in_ch)**0.5)     # Bound for uniform
        self.weight = empty(out_ch, in_ch, self.k, self.k).uniform_(-bound, bound).to(self.device)
        self.bias = empty(out_ch).uniform_(-bound, bound).to(self.device) if use_bias else empty(out_ch).fill_(0).to(self.device)
        # Initialization of the parameters' gradients
        self.grad_weight = 0*self.weight
        self.grad_bias = 0*self.bias
        
        
    def zero_grad(self):
        # Put all gradients to zero
        self.grad_weight.fill_(0.)
        if self.use_bias:
            self.grad_bias.fill_(0.)
            
    def forward(self, x):
        # Get size of interest
        self.batch_size = x.size(0)
        self.s_in = x.size(-1)
        self.s_out = int(math.ceil((x.size(-2)-self.k+1+self.padding*2)/(self.stride)))
        
        # Unfold the input to get patches of image which are touched by the kernel
        X_unf = unfold(x, kernel_size=(self.k, self.k), padding = self.padding, stride = self.stride)
        self.X_unf = X_unf
        # Matrix multiplication with kernels in lines
        O_expand = self.weight.view(self.out_ch, -1) @ X_unf
        # Retrieve the output from the result
        O = O_expand.view(self.batch_size, self.out_ch, self.s_out, self.s_out)
        # Add bias if needed
        return (O + self.bias.view(1, -1, 1, 1)) if self.use_bias else O 
    
    def backward(self, gradwrtoutput):
        # The gradwrtoutput is of size (B x OUT_CH x SO x SO)
        # The weights are of size (OUT_CH x IN_CH x SI x SI)
        # Preparation of the elements to compute the backward pass for input
        dL_dO_exp = gradwrtoutput.reshape(self.batch_size, self.out_ch, -1) # (B x OUT_CH x (SO x SO))
        dO_dX_exp = self.weight.reshape(self.out_ch,-1).transpose(0,1)      # (OUT_CH x (IN_CH x SI x SI))
        dL_dO_unf = dO_dX_exp @ dL_dO_exp                                   # (B x (IN_CH x SI x SI) x (SO x SO))
        # Retrieve the derivative w.r.t. the input using fold
        self.grad_in = fold(dL_dO_unf, kernel_size = (self.k, self.k), padding = self.padding, stride = self.stride, output_size = (self.s_in, self.s_in))
        
        # Preparation of the elements to compute the backward pass for the weights
        dL_dO_exp = gradwrtoutput.transpose(0,1).reshape(self.out_ch, -1)                           # (OUT_CH x (B x SO x SO))
        dO_dF_exp = self.X_unf.transpose(-1, -2).reshape(self.batch_size*self.s_out*self.s_out, -1) # ((B x SO x SO) x (IN_CH x K x K))
        dL_dF_exp = dL_dO_exp @ dO_dF_exp                                                           # (OUT_CH x  (IN_CH x K x K))
        # Retrieve the derivative w.r.t. the weights
        self.grad_weight = dL_dF_exp.view(self.out_ch, self.in_ch, self.k, self.k)
        
        # Preparation of the elements to compute the backward pass for the bias
        if self.use_bias:
            self.grad_bias = dL_dO_exp @ (empty(self.batch_size * (self.s_out) * (self.s_out)).fill_(1)).to(self.device)
        else:
            self.grad_bias = None
        
        return self.grad_in

    def param(self) :
        return ((self.weight, self.grad_weight), (self.bias, self.grad_bias))


############################################################################
# Transposed convolution module
class TransposeConv2d(object):
    def __init__(self, in_ch, out_ch, kernel_size = 3, padding = 0, stride = 1, use_bias = False, device='cpu', output_padding = 0):
        self.name = "TransposeConv2d"
        self.device = device
        self.use_bias = use_bias
        self.in_ch = in_ch              # Input channels
        self.out_ch = out_ch            # Output channels
        self.k = kernel_size            # Kernel size
        self.stride = stride
        self.padding = padding
        # Initialization of the parameters
        bound = 1/((self.k**2*self.in_ch)**0.5)     # Bound for uniform
        self.weight = empty(in_ch, out_ch, self.k, self.k).uniform_(-bound, bound).to(self.device)
        self.bias = empty(out_ch).uniform_(-bound, bound).to(self.device) if use_bias else empty(out_ch).fill_(0).to(self.device)
        self.output_padding = output_padding
        # Initialization of the parameters' gradients
        self.grad_weight = 0*self.weight
        self.grad_bias = 0*self.bias     
        
    def zero_grad(self):
        # Put all gradients to zero
        self.grad_weight.fill_(0.)
        if self.use_bias:
            self.grad_bias.fill_(0.)

    def forward(self, x):
        # Get size of interest
        self.x = x
        self.batch_size = x.size(0)
        # Input sizes
        self.s1 = self.x.size(-2)
        self.s2 = self.x.size(-1)
        # Output sizes
        self.o1 = (self.s1 - 1)*self.stride + 1 + self.k - 1 - self.padding *2 + self.output_padding
        self.o2 = (self.s2 - 1)*self.stride + 1 + self.k - 1 - self.padding *2 + self.output_padding
        
        # Prepare the input and kernels to obtain an unfolded copy of the output through matrix multiplication
        x_exp = x.reshape(self.batch_size, self.in_ch, -1)
        K_exp = self.weight.reshape(self.in_ch,-1).transpose(0,1)
        # Retrieve the output using fold
        out = fold(K_exp @ x_exp, kernel_size = (self.k, self.k), padding = self.padding, stride = self.stride, output_size = (self.o1,self.o2))
        # Add the bias if needed
        return (out + self.bias.view(1, -1, 1, 1)) if self.use_bias else out
    
    def backward(self, gradwrtoutput):
        # gradwrtoutput is of size (B x OUT_CH x SO x SO)
        # Unfold the gradwrtoutput to get patches of image which are touched by the kernel
        dL_dO_unf = unfold(gradwrtoutput, kernel_size = (self.k, self.k), padding = self.padding, stride = self.stride) # B x (OUT_CH x K x K) x SI x SI
        dL_dX_exp = self.weight.view(self.in_ch, -1) @ dL_dO_unf
        # Retrieve the result from the matrix multiplication of the kernel with the unfolded gradwrtoutput
        self.grad_in = dL_dX_exp.view(self.batch_size, self.in_ch, self.s1, self.s2)
        
        # Preparation of the elements to compute the backward pass for the weights
        dL_dO_unf_K = dL_dO_unf.transpose(0,1).reshape(self.out_ch * self.k * self.k, -1).transpose(0,1) # (B x SI x SI) x (OUT_CH x K x K)
        dO_dF_exp = self.x.transpose(0,1).reshape(self.in_ch, -1)                                        # IN_CH x (B x SI x SI)
        dL_dF_exp = dO_dF_exp @ dL_dO_unf_K                                                              # IN_CH x (OUT_CH x K x K)                 # Retrieve the derivative w.r.t. the weights
        self.grad_weight = dL_dF_exp.view(self.in_ch, self.out_ch, self.k, self.k)                       # OUT_CH x IN_CH x K x K
        
        # Preparation of the elements to compute the backward pass for the bias
        if self.use_bias:
            dO_dB_exp = (empty(self.batch_size * (self.o1) * (self.o2)).fill_(1)).to(self.device)
            self.grad_bias = gradwrtoutput.transpose(0,1).reshape(self.out_ch, -1) @ dO_dB_exp
        else:
            self.grad_bias = None
        
        return self.grad_in
        
    def param(self) :
        return ((self.weight, self.grad_weight), (self.bias, self.grad_bias))
