import torch 
from tqdm import tqdm 

from utils import *
        
class Model():
    def __init__(self,device = 'cpu'):
        self.device = device
        self.batch_size = 100

        self.in_ch = 3
        self.m = 100
        self.k = 3
        self.lr = 1e-2
        stride = 2
        padding = 1
        output_padding = 1
        use_bias=True
    
        self.conv1 = Conv2d(self.in_ch,self.m, kernel_size = (self.k,self.k), stride=stride, padding=padding, device=device, use_bias = use_bias)
        self.conv2 = Conv2d(self.m, 2*self.m, kernel_size = (self.k,self.k), stride=stride, padding=padding,device=device, use_bias = use_bias)
        self.tconv1 = ConvTranspose2d(2*self.m, self.m, kernel_size = (self.k,self.k), stride=stride, padding=padding, device=device, output_padding=output_padding, use_bias = use_bias)
        self.tconv2 = ConvTranspose2d(self.m, self.in_ch, kernel_size = (self.k,self.k), stride=stride, padding=padding, device=device, output_padding=output_padding, use_bias = use_bias)   


        self.model = Sequential(self.conv1,ReLU(),
            self.conv2,ReLU(),
            self.tconv1,ReLU(),
            self.tconv2,Sigmoid()
            )
        
        # optimizer
        self.optimizer = SGD(self.model, lr = self.lr)
        self.optimizer.set_momentum(0.9)
        
        # loss
        self.mse = MSE()
        
    def train(self,noisy_imgs_1, noisy_imgs_2, n_epochs):
        noisy_imgs_1 = noisy_imgs_1.to(self.device).float()/256
        noisy_imgs_2 = noisy_imgs_2.to(self.device).float()/256
        inp_1 = noisy_imgs_1.clone().type(torch.float).split(self.batch_size)
        tar_1 = noisy_imgs_2.clone().type(torch.float).split(self.batch_size)
        
        self.loss_train = empty(n_epochs).fill_(0).to(self.device)
        for e in tqdm(range(n_epochs)):
            for i in (range(len(inp_1))):
                out = self.model.forward(inp_1[i])
                loss = self.mse.forward(out, tar_1[i])
                self.optimizer.zero_grad()
                out_grad = self.model.backward(self.mse.backward())
                self.optimizer.step() 
                self.loss_train[e] += loss

    
    def predict(self,noisy_imgs):
        noisy_imgs = noisy_imgs.clone().to(self.device).float()/256
        out = self.model.forward(noisy_imgs)
        return out
