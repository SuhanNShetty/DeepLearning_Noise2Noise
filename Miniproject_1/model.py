import torch 
from torch import nn
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self,in_ch, m, k):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, m, kernel_size = k, stride=1, padding=0)
        self.conv2 = nn.Conv2d(m, m*2, kernel_size = k, stride=1, padding=0)
        self.tconv1 = nn.ConvTranspose2d(m*2, m, kernel_size = k, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(m, in_ch, kernel_size = k, stride=1, padding=0)
        
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
        self.batch_size, self.nb_epochs = 50, 1000
        self.in_ch = 3
        self.m = 16
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
        input = train_input.to(self.device).type(torch.float).split(self.batch_size)
        target = train_target.to(self.device).type(torch.float).split(self.batch_size)
#         valid_inp = noisy_imgs.to(device).type(torch.float)
#         valid_target = clean_imgs.to(device).type(torch.float)
        
        loss_train = torch.zeros(num_epochs, device = self.device)
        loss_valid = torch.zeros(num_epochs, device = self.device)
        
        for e in tqdm(range(num_epochs)):
            
            self.model.train()
            for i in range(len(input)):
                output = self.model(input[i])
                loss_batch = self.mse(output, target[i])
                loss_train[e] += loss_batch
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

    def train_with_scheduler(self, train_input, train_target, valid_input, valid_target, num_epochs):
        input = train_input.to(self.device).type(torch.float).split(self.batch_size)
        target = train_target.to(self.device).type(torch.float).split(self.batch_size)
        valid_inp = valid_input.to(self.device).type(torch.float)
        valid_target = valid_target.to(self.device).type(torch.float)
        
        loss_train = torch.zeros(num_epochs, device = self.device)
        loss_valid = torch.zeros(num_epochs, device = self.device)
        
        for e in tqdm(range(num_epochs)):
            
            self.model.train()
            for i in range(len(input)):
                output = self.model(input[i])
                loss_batch = self.mse(output, target[i])
                loss_train[e] += loss_batch
                loss_batch.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                
            self.model.eval()
            output = self.model(valid_inp)
            loss_valid[e] = self.mse(valid_inp, valid_target)
            
            self.scheduler.step(loss_valid[e])
            
    def predict(self, test_input):
        
        return self.model(test_input.to(self.device))