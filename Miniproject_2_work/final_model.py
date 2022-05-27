from torch.cuda import is_available
from tqdm import tqdm 
from torch import save, load

from utils import *
        
class Model():
    def __init__(self):
        # Find the device to use
        self.device = 'cuda' if is_available() else 'cpu'
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
        
        # Modules to be used, 2 Conv2d and 2 Upsampling or TransposeConv2d
        self.conv1 = Conv2d(self.in_ch,self.m, kernel_size = (self.k,self.k), stride=stride, padding=padding, device=self.device, use_bias = use_bias)
        self.conv2 = Conv2d(self.m, 2*self.m, kernel_size = (self.k,self.k), stride=stride, padding=padding, device=self.device, use_bias = use_bias)
        self.tconv1 = TransposeConv2d(2*self.m, self.m, kernel_size = (self.k,self.k), stride=stride, padding=padding, device=self.device, output_padding=output_padding, use_bias = use_bias)
        self.tconv2 = TransposeConv2d(self.m, self.in_ch, kernel_size = (self.k,self.k), stride=stride, padding=padding, device=self.device, output_padding=output_padding, use_bias = use_bias)
        
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
        
    def train(self,noisy_imgs_1, noisy_imgs_2, n_epochs):
        # Retrieve images and prepare them
        noisy_imgs_1 = noisy_imgs_1.to(self.device).float()/256
        noisy_imgs_2 = noisy_imgs_2.to(self.device).float()/256
        inp_1 = noisy_imgs_1.clone().split(self.batch_size)
        tar_1 = noisy_imgs_2.clone().split(self.batch_size)
        # Prepare a torch to retrieve the loss during training
        self.loss_train = empty(n_epochs).fill_(0).to(self.device)
        for e in tqdm(range(n_epochs)):
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
        return self.model.forward(noisy_imgs)
    
    def save_model(self):
        path = 'test_model.pth'
        save(self.model, path)
        pass
    
    def load_model(self):
        path = 'test_model.pth'
        self.model = load(path, map_location = self.device)
        pass
