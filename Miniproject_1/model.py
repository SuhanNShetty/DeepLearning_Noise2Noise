import torch 
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Sigmoid
from tqdm import tqdm 

class Net(nn.Module):
    def __init__(self,in_ch, m, k):
        super().__init__()

        self.Block = nn.Sequential(Conv2d(in_ch,m, kernel_size=k, stride=2, padding=1), ReLU(),
                                   ConvTranspose2d(m, in_ch, kernel_size=k, stride=2, padding=1, output_padding=1), Sigmoid())

        # self.Block = nn.Sequential(Conv2d(in_ch,m, kernel_size=k, stride=2, padding=1), ReLU(),
        #                            Conv2d(m,m, kernel_size=k, stride=2, padding=1), ReLU(),
        #                            ConvTranspose2d(m, m, kernel_size=k, stride=2, padding=1, output_padding=1), ReLU(),
        #                            ConvTranspose2d(m, in_ch, kernel_size=k, stride=2, padding=1, output_padding=1), Sigmoid())


#         self.Block = nn.Sequential(Conv2d(in_ch,m, kernel_size=k, stride=2, padding=1), ReLU(),
#                                    Conv2d(m,m, kernel_size=k, stride=2, padding=1), ReLU(),
#                                    Conv2d(m,m, kernel_size=k, stride=2, padding=1), ReLU(),
#                                    Conv2d(m,m, kernel_size=k, stride=2, padding=1), ReLU(),
#                                    ConvTranspose2d(m, m, kernel_size=k, stride=2, padding=1, output_padding=1), ReLU(),
#                                    ConvTranspose2d(m, m, kernel_size=k, stride=2, padding=1, output_padding=1), ReLU(),
#                                    ConvTranspose2d(m, m, kernel_size=k, stride=2, padding=1, output_padding=1),ReLU(),
#                                    ConvTranspose2d(m, in_ch, kernel_size=k, stride=2, padding=1, output_padding=1), Sigmoid())
        
    def forward(self, x):
        return self.Block(x)
        
class Model():
    def __init__(self, device='cpu'):
        self.device=device
        self.batch_size = 1000
        self.in_ch = 3
        self.m = 200
        self.k = 3
        
        # Instantiate model
        self.model = Net(self.in_ch, self.m, self.k)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-2) #SGD(self.model.parameters(), lr = 1, momentum=0.9)#
        
        # Loss function
        self.mse = nn.MSELoss()
        
        # # Scheduler
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, threshold  = 1e-10)
        
    def load_pretrained_model(self):
        torch.load(self.model,'bestmodel.pth')
    
    def train(self, train_input, train_target, num_epochs):
        train_input  = (train_input.float()/256).to(self.device)
        train_target = (train_target.float()/256).to(self.device)
        
        train_input  = train_input.type(torch.float).split(self.batch_size)
        train_target = train_target.type(torch.float).split(self.batch_size)
        
        # split the training set in a training and validation set 
        split = torch.floor(torch.tensor(len(train_input)*0.9)).int().item()
        
        input = train_input[0:split]
        valid_input = train_input[split:-1]
        
        target = train_target[0:split]
        valid_target = train_target[split:-1]
        
        self.loss_train = torch.zeros(num_epochs, device = self.device,requires_grad=False)
        self.loss_valid = torch.zeros(num_epochs, device = self.device,requires_grad=False)
        
        for e in tqdm(range(num_epochs)):
            self.model.train()
            for i in range(len(input)):
                output = self.model(input[i])
                loss_batch = self.mse(output, target[i])
                self.loss_train[e] += loss_batch.item()
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()
            
            self.model.eval()
            for j in range(len(valid_input)): 
                output = self.model(valid_input[j])
                loss_batch = self.mse(output, target[i])
                self.loss_valid[e] += loss_batch.item()
#             self.scheduler.step(self.loss_valid[e])

                                        

    def predict(self, test_input):
        self.model.eval()
        test_input = test_input.float().to(self.device).type(torch.float)/256
        test_output = self.model(test_input)*256
        return test_output.type(torch.ByteTensor)