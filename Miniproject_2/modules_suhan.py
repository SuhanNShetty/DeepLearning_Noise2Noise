import torch

class SGD:
	'''
		Gets a tuple of ((weight,grad_weight),(bias,grad_bias)) and updates the learning parameters
		Disclaimer: We used https://github.com/pytorch/pytorch/blob/cd9b27231b51633e76e28b6a34002ab83b0660fc/torch/optim/sgd.py#L63
		as reference for writing our version
	'''
	def __init__(self, transforms, lr=1e-6, use_momentum=False, damping=0.):
		self.params = params # params[i]: ((w,grad_w),(b,grad_b))
		self.lr = lr
		# momentum params
		self.damping = damping
		self.transforms = transforms
		if use_momentum is True:
			self.set_velocity()
			if damping < 0 :
				raise ValueError("Damping can not be < 0 for SGD with momentum")	

	def set_velocity(self):
		'''
			Initialize velocity for momentum
		'''
		self.velocity = []
		for tfm in self.transforms:
			self.velocity.append((0*tfm.params[0][1],0*tfm.params[1][1]))

	def zero_grad(self):
		'''
			Set gradients to zero
		'''
		for tfm in self.transforms:
			tfm.params[0][1] = 0*tfm.params[0][1]
			tfm.params[1][1] = 0*tfm.params[1][1]

	def step(self):
		'''
			Take one gradient step 
		'''
		for i,tfm in enumerate(self.transforms):
			# set momentum
			self.velocity[i][0] = self.damping*self.velocity[i][0] + self.params[i][0][1] # weight
			self.velocity[i][1] = self.damping*self.velocity[i][1] + self.params[i][1][1]  # bias
			# update weight and bias:
			tfm.params[0][0] -= self.lr*self.velocity[i][0] # weight
			tfm.params[1][0] -= self.lr*self.velocity[i][1] # bias
			# zero the grad:
			tfm.params[0][1] = 0*tfm.params[0][1] # weight
			tfm.params[1][1] = 0*tfm.params[1][1] # bias


class Sequential:
	def __init__(self, *args):
		self.params = []
		self.set_sequential(args)

	def forward(self, x, params):
		for tfm in self.transforms:
			tfm.set_params(tfm.params[0][0])
			x = tfm.forward(x,weight=tfm.params[0][0],bias=tfm.params[1][0])
			self.params
		return x

	def backward(self, grad_out):
		'''
			Collect the gradient by backpropogation 
		'''
		for tfm in self.transforms[::-1]:
			grad_out, grad_params = tfm.backward(grad_out)	
			tfm.params[0][1] = grad_params[0] # To do: use chain rule
			tfm.params[1][1] = grad_params[1]
		return grad_out



class LinearLayer:
	'''
		y = xA' + b
	'''
	def __init__(self, in_features, out_features, bias=True, device=None):
		self.weight = torch.empty(in_features,out_features).randn_().to(device)
		self.bias = torch.zeros((1,out_features)).to(device)
		self.params = ((self.weight,0*self.weight),(self.bias,0*self.bias))

	def forward(self,x):
		y = x@self.weight.flip(0,1) + self.bias
		return y 

	def backward(self,grad_param_next):
		# return the output of chain rule
		pass