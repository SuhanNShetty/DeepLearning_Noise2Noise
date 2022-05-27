import torch 
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Sigmoid, MaxPool2d
from tqdm import tqdm
from pathlib import Path

class Net(nn.Module):
    def __init__(self, in_ch = 3, m = 16, k = 3):
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
        self.tconv4 = ConvTranspose2d(m*2, m, kernel_size = k, stride=stride, padding=1, output_padding=output_padding)
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
        x10 = self.sigmoid(self.tconv5(torch.cat((x9,x1),1)))
        return x10

class Model():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 100
        self.in_ch = 3
        self.m = 32
        self.k = 3
        
        # Model
        self.model = Net(in_ch = self.in_ch, m = self.m, k = self.k)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr) #SGD(self.model.parameters(), lr = 1, momentum=0.9)#
        
        # Loss function
        self.mse = nn.MSELoss()
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, threshold  = 1e-10)
        
    def load_pretrained_model(self):
        f = Path('bestmodel.pth')
        p = Path('Miniproject_2')
        path = p / f if Path.is_dir(p) else f
        torch.load(self.model,path)
        
    def train(self,noisy_imgs_1, noisy_imgs_2, n_epochs):
        train_input  = (train_input.float()/256).to(self.device)
        train_target = (train_target.float()/256).to(self.device)
        
        train_input  = train_input.split(self.batch_size)
        train_target = train_target.split(self.batch_size)
        
        # Split the training set in a training and validation set
        split = torch.floor(torch.tensor(len(train_input)*0.9)).int().item()
        
        input = train_input[0:split]
        valid_input = train_input[split:-1]
        
        target = train_target[0:split]
        valid_target = train_target[split:-1]
        
        self.loss_train = torch.zeros(num_epochs, device = self.device,requires_grad=False)
        self.loss_valid = torch.zeros(num_epochs, device = self.device,requires_grad=False)
        
        for e in range(num_epochs):
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

    
    def predict(self,noisy_imgs):
        # Retrieve images and prepare them
        noisy_imgs = noisy_imgs.clone().to(self.device).float()/256
        return (self.model(noisy_imgs)*256).byte()
    
    def save_model(self):
        f = Path('bestmodel.pth')
        p = Path('Miniproject_2')
        path = p / f if Path.is_dir(p) else f

        save(self.model, path)
        
    def load_pretrained_model(self):
        f = Path('bestmodel.pth')
        p = Path('Miniproject_2')
        path = p / f if Path.is_dir(p) else f

        self.model = load(path, map_location = self.device)
        

