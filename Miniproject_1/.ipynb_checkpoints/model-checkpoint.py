import torch 
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Sigmoid, MaxPool2d
from tqdm import tqdm 
from torch.profiler import profile, record_function, ProfilerActivity


# class Net(nn.Module):
#     def __init__(self,in_ch, m, k):
#         super().__init__()

#         self.Block = nn.Sequential(Conv2d(in_ch,m, kernel_size=k, stride=1, padding=1), ReLU(),
#                                    ConvTranspose2d(m, in_ch, kernel_size=k, stride=1, padding=1, output_padding=1), Sigmoid())

        
#     def forward(self, x):
#         return self.Block(x)



# class Net(nn.Module):
#     def __init__(self,in_ch, m, k):
#         super().__init__()

# #         self.Block = nn.Sequential(Conv2d(in_ch,m, kernel_size=k, stride=2, padding=1), ReLU(),
# #                                    ConvTranspose2d(m, in_ch, kernel_size=k, stride=2, padding=1, output_padding=1), Sigmoid())

# #         self.Block = nn.Sequential(Conv2d(in_ch,m, kernel_size=k, stride=2, padding=1), ReLU(),
# #                                    Conv2d(m,m, kernel_size=k, stride=2, padding=1), ReLU(),
# #                                    ConvTranspose2d(m, m, kernel_size=k, stride=2, padding=1, output_padding=1), ReLU(),
# #                                    ConvTranspose2d(m, in_ch, kernel_size=k, stride=2, padding=1, output_padding=1), Sigmoid())


#         self.BlockConv1 = nn.Sequential(Conv2d(in_ch,m, kernel_size=k, stride=1, padding=1), ReLU(),
#                                    Conv2d(m,m, kernel_size=k, stride=1, padding=1), ReLU())
    
#         self.BlockConv2 = nn.Sequential(Conv2d(m,m, kernel_size=k, stride=1, padding=1), ReLU(),
#                                    Conv2d(m,m, kernel_size=k, stride=1, padding=1), ReLU())

#         self.BlockDConv1 = nn.Sequential(ConvTranspose2d(m, m, kernel_size=k, stride=2, padding=1, output_padding=1),ReLU(),
#                                     ConvTranspose2d(m, m, kernel_size=k, stride=2, padding=1, output_padding=1), ReLU())

#         self.BlockDConv2 = nn.Sequential(ConvTranspose2d(2*m, m, kernel_size=k, stride=2, padding=1, output_padding=1),ReLU(),
#                                     ConvTranspose2d(2*m, in_ch, kernel_size=k, stride=2, padding=1, output_padding=1), Sigmoid())


#     def forward(self,x):
#         x1 = self.BlockConv1(x)
#         x2 = self.BlockConv2(x1)
#         x3 = self.BlockDConv1(x2)
#         x4 = self.BlockConv2(torch.cat((x1,x3),dim=1))
#         return x4
        
#     def forward(self, x):
#         return self.Block(x)



class Net(nn.Module):
    def __init__(self,in_ch, m, k):
        super().__init__() 
        stride = 1
        output_padding = 0
        self.conv1 = Conv2d(in_ch,m, kernel_size = k, stride=stride, padding=1)
        self.conv2 = Conv2d(m, m, kernel_size = k, stride=stride, padding=1)
        self.conv3 = Conv2d(m, m, kernel_size = k, stride=stride, padding=1)
        self.conv4 = Conv2d(m, m, kernel_size = k, stride=stride, padding=1)
        self.conv5 = Conv2d(m, m, kernel_size = k, stride=stride, padding=1)
        
        self.tconv1 = ConvTranspose2d(m, m, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
        self.tconv2 = ConvTranspose2d(m*2, m, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
        self.tconv3 = ConvTranspose2d(m*2, m, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
        self.tconv4 = ConvTranspose2d(m*2, 2*m, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
        self.tconv5 = ConvTranspose2d(m*2, in_ch, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.tconv1(x3))
        x7 = self.relu(self.tconv2(torch.cat((x6,x4),1)))
        x8 = self.relu(self.tconv3(torch.cat((x7,x3),1)))
        x9 = self.relu(self.tconv4(torch.cat((x8,x2),1)))
        x10 = self.sigmoid(self.tconv5(x9))
        return x10







# class Net(nn.Module):
#     def __init__(self,in_ch, m, k):
#         super().__init__() 
#         stride = 1
#         output_padding = 0
#         self.conv1 = Conv2d(in_ch,m, kernel_size = k, stride=stride, padding=1)
#         self.conv2 = Conv2d(m, m*2, kernel_size = k, stride=stride, padding=1)
#         self.conv3 = Conv2d(m*2, m*2, kernel_size = k, stride=stride, padding=1)
#         self.tconv0 = ConvTranspose2d(m*2, m*2, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
#         self.tconv1 = ConvTranspose2d(m*4, m, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
#         self.tconv2 = ConvTranspose2d(m*2, in_ch, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
        
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         x1 = self.relu(self.conv1(x))
#         x2 = self.relu(self.conv2(x1))
#         x3 = self.relu(self.conv3(x2))
#         x3 = self.relu(self.tconv0(x3))
#         x3 = self.relu(self.tconv1(torch.cat((x3,x2),1)))
#         x3 = self.sigmoid(self.tconv2(torch.cat((x3,x1),1)))
#         return x3

        
class Model():
    def __init__(self, device='cpu'):
        self.device=device
        
        self.batch_size = 1000
        self.in_ch = 3
        self.m = 10
        self.k = 3
        
        # Instantiate model
        self.model = Net(self.in_ch, self.m, self.k)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-2) #SGD(self.model.parameters(), lr = 1, momentum=0.9)#
        
        # Loss function
        self.mse = nn.MSELoss()
        
        # # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, threshold  = 1e-10)
        
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
            self.scheduler.step(self.loss_valid[e])

                                        

    def predict(self, test_input):
        self.model.eval()
        test_input = test_input.float().to(self.device).type(torch.float)/256
        test_output = self.model(test_input)*256
        return test_output.type(torch.ByteTensor)