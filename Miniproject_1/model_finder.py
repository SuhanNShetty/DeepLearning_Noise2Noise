import torch
torch.manual_seed(10)

from model import Model
import matplotlib.pyplot as plt
import torchvision.transforms as T 
from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--net', type=int, default=0) # choose a model: net in (2,4,6,8)  #layers
args = parser.parse_args()

file_name = "Net-"+str(args.net) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


path_train = '../data/train_data.pkl'
path_val = '../data/val_data.pkl'

noisy_imgs_1, noisy_imgs_2 = torch.load(path_train)
noisy_imgs , clean_imgs = torch.load(path_val)

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y)**2).mean((1,2,3))).mean()


# Select hyper-params: batchsize and number of channels in the model
psnr_m_bs = []
m_all = [10] #[10,25,50,100]
bs_all =  [10]#[10,50,100,250,500,1000]
loss_train_m_bs = []
loss_val_m_bs = []
for m in m_all:
    psnr_m = []
    loss_train_m = []
    loss_val_m = []
    for bs in bs_all:
        n_epochs = 1+ int(0.25*bs/10)
        print("Model Training for m:{}, bs:{}, epochs:{}".format(m,bs,n_epochs))
        model = Model(bs=bs,m=m, net=args.net, lr=1e-2, device=device)
        model.train(noisy_imgs_1, noisy_imgs_2, n_epochs)
        out = model.predict(noisy_imgs)
        psnr_ = compute_psnr(out.cpu().float()/256, clean_imgs.float()/256)
        psnr_m.append(psnr_)
        print("m:{}, bs:{}, psnr:{}".format(m,bs,psnr_))
        loss_train_m.append(model.loss_train)
        loss_val_m.append(model.loss_valid)
    psnr_m_bs.append(psnr_m)
    loss_train_m_bs.append(loss_train_m)
    loss_val_m_bs.append(loss_val_m)
torch.save({'psnr': psnr_m_bs, 'bs':bs_all, 'm':m_all, 'loss_train':loss_train_m_bs,'loss_val':loss_val_m_bs},file_name+ ".pickle")


psnr_t = torch.tensor(psnr_m_bs)
print("psnr: ", psnr_t)
p = torch.argmax(psnr_t)
i = torch.floor(p/psnr_t.shape[1]).long()
j = (p-i*psnr_t.shape[1]).long()
m = m_all[i.item()]
bs = bs_all[j.item()]

torch.save({'psnr': psnr_m_bs, 'bs':bs_all, 'm':m_all, 'best_m':m, 'best_bs':bs, 'loss_train':loss_train_m_bs,'loss_val':loss_val_m_bs},file_name+ ".pickle")

# Choose Learning Rate
print("Choose Learning Rate: ")
lr_all = [1e-1,5e-2, 1e-2, 5e-3, 1e-3]
psnr_lr = []
loss_train_lr = []
loss_val_lr = []
for lr in lr_all:
    n_epochs = 1+ int(0.25*bs/10)
    print("Model Training for m:{}, bs:{}, epochs:{}".format(m,bs,n_epochs))
    model = Model(bs=bs, m=m, net=args.net, lr=lr, device=device)
    model.train(noisy_imgs_1, noisy_imgs_2, n_epochs)
    out = model.predict(noisy_imgs)
    psnr_ = compute_psnr(out.cpu().float()/256, clean_imgs.float()/256)
    psnr_lr.append(psnr_)
    loss_train_lr.append(model.loss_train)
    loss_valid_lr.append(model.loss_valid)
    print("m:{}, bs:{}, lr:{}, psnr:{}".format(m,bs,lr,psnr_))  
print("psnr over lr: ", psnr_lr)
lr = lr_all[torch.argmax(torch.tensor(psnr_lr))]
print("Best Learning Rate: ", lr)

torch.save({'psnr': psnr_m_bs, 'bs':bs_all, 'm':m_all, 'best_m':m, 'best_bs':bs, 'psnr_lr':psnr_lr, 'best_lr':lr, 'loss_train':loss_train_m_bs,'loss_val':loss_val_m_bs, 'loss_train_lr':loss_train_lr,'loss_valid_lr':loss_valid_lr},file_name+ ".pickle")


# Train the model with the best hyper-params:

n_epochs = 1+ 20*int(0.25*bs/10)

print("Train the best model, m:{}, bs:{}, lr:{}, epochs:{}".format(m,bs,lr,n_epochs))

model = Model(bs=bs,m=m, net=args.net, lr=lr, device=device)
model.train(noisy_imgs_1, noisy_imgs_2, n_epochs)
out = model.predict(noisy_imgs)
psnr_ = compute_psnr(out.cpu().float()/256, clean_imgs.float()/256)
print("Best psnr: ", psnr_)

torch.save({'psnr': psnr_m_bs, 'bs':bs_all, 'm':m_all, 'best_m':m, 'best_bs':bs, 'psnr_lr':psnr_lr, 'best_lr':lr, 'best_psnr':psnr_, 'loss_train':loss_train_m_bs,'loss_val':loss_val_m_bs, 'loss_train_lr':loss_train_lr,'loss_valid_lr':loss_valid_lr, 'loss_train_best':model.loss_train,'loss_val_best':model.loss_valid, },file_name+ ".pickle")


