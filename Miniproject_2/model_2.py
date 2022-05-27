import torch
from torch import empty , cat , arange, device
from torch.nn.functional import fold , unfold
from tqdm import tqdm 
from utils import *
        
class Model():
    def __init__(self):
        self.device = device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 50
        self.in_ch = 3
        self.m = 32
        self.k = 3
        
        # Instantiate elements of the model
        self.conv1 = Conv2d(self.in_ch,self.m, kernel_size = (self.k,self.k), stride=1, padding=0, device=self.device, use_bias = False)
        self.conv2 = Conv2d(self.m, 2*self.m, kernel_size = (self.k,self.k), stride=1, padding=0,device=self.device, use_bias = False)
        self.tconv1 = ConvTranspose2d(2*self.m, self.m, kernel_size = (self.k,self.k), stride=1, padding=0,device=self.device, use_bias = False)
        self.tconv2 = ConvTranspose2d(self.m, self.in_ch, kernel_size = (self.k,self.k), stride=1, padding=0,device=self.device, use_bias = False)        
        
        # Setup the model
        self.model = Sequential(self.conv1,ReLU(),
                            self.conv2,ReLU(),
                            self.tconv1,ReLU(),
                            self.tconv2,Sigmoid())
        # Optimizer
        self.optimizer = SGD(self.model, lr=1, use_momentum=False, momentum=0.5) 

        # Loss function
        self.mse = MSE()
        
    def save_model(self):
        # to do: save as pckle file
        pass

    def load_pretrained_model(self):
        # To do: load pretrained model - a pickle file
        pass
        #torch.load(self.model,'bestmodel.pth')
    
    def train(self, train_input, train_target, num_epochs):
        train_input  = train_input.float()/256
        train_target = train_target.float()/256
        
        input  = train_input.to(self.device).type(torch.float).split(self.batch_size)
        target = train_target.to(self.device).type(torch.float).split(self.batch_size)
        
        self.loss_train = 0*empty(num_epochs, device = self.device, requires_grad=False)
        
        for e in tqdm(range(num_epochs)):
            for i in range(len(input)):
                output = self.model.forward(input[i])
                loss_batch = self.mse.forward(output, target[i])
                self.loss_train[e] += loss_batch.item()
                self.optimizer.zero_grad()
                self.model.backward(self.mse.backward()) # update the gradients
                self.optimizer.step()
                                                                        

    def predict(self, test_input):
        test_input = test_input.float().to(self.device).type(torch.float)/256
        test_output = self.model.forward(test_input)*256
        return test_output.long()
