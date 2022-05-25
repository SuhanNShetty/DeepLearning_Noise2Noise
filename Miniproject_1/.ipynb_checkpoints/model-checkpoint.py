import torch 
from torch import nn
from tqdm import tqdm 

class Net(nn.Module):
    def __init__(self,in_ch, m, k):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_ch,m, kernel_size = k, stride=2, padding=1)
        self.conv2 = nn.Conv2d(m, m*2, kernel_size = k, stride=2, padding=1)
        self.tconv1 = nn.ConvTranspose2d(m*2, m, kernel_size = k, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(m, in_ch, kernel_size = k, stride=2, padding=1, output_padding=1)
        
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
    def __init__(self, device='cpu'):
        self.device=device
        self.batch_size = 100
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
        
        # # Scheduler
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, threshold  = 1e-10)
        
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
            m1_old = torch.cuda.memory_allocated()
            for i in range(len(input)):
                print(torch.cuda.memory_allocated()-m1_old)
                mq_old  = torch.cuda.memory_allocated()
                print('######')
                m2 = torch.cuda.memory_allocated()
                print('line 1: ',m2-m1)
                output = self.model(input[i])
                m3 = torch.cuda.memory_allocated()                
                print('line 2: ',m3-m2)
                loss_batch = self.mse(output, target[i])
                m4 = torch.cuda.memory_allocated()
                print('line 3: ',m4-m3)
                self.loss_train[e] += loss_batch
                m5 = torch.cuda.memory_allocated()
                print('line 4: ',m5-m4)
                self.optimizer.zero_grad()
                m6 = torch.cuda.memory_allocated()
                print('line 5: ',m6-m5)
                loss_batch.backward()
                m7 = torch.cuda.memory_allocated()
                print('line 6: ',m7-m6)
                self.optimizer.step()
                m1 = torch.cuda.memory_allocated()
                print('line 7: ',m1-m7)
            
            self.model.eval()
            m1 = torch.cuda.memory_allocated()
            for j in range(len(valid_input)): 
                output = self.model(valid_input[j])
                m2 = torch.cuda.memory_allocated()
                print('line 8: ', m2-m1)
                loss_batch = self.mse(output, target[i])
                m3 = torch.cuda.memory_allocated()
                print('line 9: ', m3-m2)
                self.loss_valid[e] += loss_batch
                m1 = torch.cuda.memory_allocated()
                print('line 10: ', m1-m3)
#             torch.cuda.empty_cache()
                                    
#             self.scheduler.step(self.loss_valid[e])
                                    

    def predict(self, test_input):
        self.model.eval()
        test_input = test_input.float().to(self.device).type(torch.float)/256
        test_output = self.model(test_input)*256
        return test_output.type(torch.ByteTensor)