import torch 
from tqdm import tqdm 
from modules_suhan import *
        
class Model():
    def __init__(self, device):
        self.device = device
        self.batch_size = 50
        self.in_ch = 3
        self.m = 32
        self.k = 3
        
        # Instantiate elements of the model
        self.conv1 = Conv2d(self.in_ch,self.m, kernel_size = (self.k,self.k), stride=1, padding=0, device=device)
        self.conv2 = Conv2d(self.m, 2*self.m, kernel_size = (self.k,self.k), stride=1, padding=0,device=device)
        self.tconv1 = ConvTranspose2d(2*self.m, self.m, kernel_size = (self.k,self.k), stride=1, padding=0,device=device)
        self.tconv2 = ConvTranspose2d(self.m, self.in_ch, kernel_size = (self.k,self.k), stride=1, padding=0,device=device)        
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

        # Setup the model
        self.model = Sequential(self.conv1,self.relu,
                            self.conv2,self.relu,
                            self.tconv1, self.relu,
                            self.tconv2, self.sigmoid)
        # Optimizer
        self.optimizer = SGD(self.model,lr=1e-3, use_momentum=False, damping=0.) 

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
        
        train_input  = train_input.to(self.device).type(torch.float).split(self.batch_size)
        train_target = train_target.to(self.device).type(torch.float).split(self.batch_size)
        
        # split the training set in a training and validation set 
        split = torch.floor(torch.tensor(len(train_input)/10*9)).int().item()
        
        input = train_input[0:split]
        valid_input = train_input[split:-1]
        
        target = train_target[0:split]
        valid_target = train_target[split:-1]
        
        self.loss_train = torch.zeros(num_epochs, device = self.device)
        self.loss_valid = torch.zeros(num_epochs, device = self.device)
        
        for e in tqdm(range(num_epochs)):
            for i in range(len(input)):
                output = self.model.forward(input[i])
                loss_batch = self.mse.forward(output, target[i])
                self.loss_train[e] += loss_batch
                self.optimizer.zero_grad()
                self.model.backward(self.mse.backward()) # update the gradients
                self.optimizer.step()  
            print("loss:",self.loss_train[e])    
            for j in range(len(valid_input)):
                output = self.model.forward(valid_input[j])
                loss_batch = self.mse.forward(output, target[i])
                self.loss_valid[e] += loss_batch
                                                                        

    def predict(self, test_input):
        test_input = test_input.float().to(self.device).type(torch.float)/256
        test_output = self.model.forward(test_input)*256
        return test_output.long()
