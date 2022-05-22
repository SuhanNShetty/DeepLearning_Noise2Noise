import torch 
from torch import nn
from tqdm import tqdm 

class Net(nn.Module):
    def __init__(self,in_ch, m, k):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_ch,m, kernel_size = k, stride=1, padding=0)
        self.conv2 = nn.Conv2d(m, m*2, kernel_size = k, stride=1, padding=0)
        self.tconv1 = nn.ConvTranspose2d(m*2, m, kernel_size = k, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(m, in_ch, kernel_size = k, stride=1, padding=0)
        
        self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.tconv1(x))
        x = self.sigmoid(self.tconv2(x))
        return x
        
class Model():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 50
        self.in_ch = 3
        self.m = 32
        self.k = 3
        
        # Instantiate model
        self.model = Net(self.in_ch, self.m, self.k)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-2)
        
        # Loss function
        self.mse = nn.MSELoss()
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, threshold  = 1e-10)
        
    def load_pretrained_model(self):
        torch.load(self.model,'bestmodel.pth')
    
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
            
            self.model.train()
            for i in range(len(input)):
                output = self.model(input[i])
                loss_batch = self.mse(output, target[i])
                self.loss_train[e] += loss_batch
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()
                
            self.model.eval()
            for j in range(len(valid_input)):
                output = self.model(valid_input[j])
                loss_batch = self.mse(output, target[i])
                self.loss_valid[e] += loss_batch
                                    
            self.scheduler.step(self.loss_valid[e])
                                    

    def predict(self, test_input):
        self.model.eval()
        test_input = test_input.float().to(self.device).type(torch.float)/256
        test_output = self.model(test_input)*256
        return test_output.type(torch.ByteTensor)