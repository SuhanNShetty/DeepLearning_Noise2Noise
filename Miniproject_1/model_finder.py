import torch
torch.manual_seed(10)

from model import Model
import matplotlib.pyplot as plt
import torchvision.transforms as T 
from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--net', type=int, default=0)
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

psnr = []
m_all = [10,25,50,100]
bs_all = [10,50,100,250,500,1000]
for m in m_all:
    psnr_m = []
    for bs in bs_all:
        n_epochs = 2*int(1+bs/10)
        print("Model Training for m:{}, bs:{}, epochs:{}".format(m,bs,n_epochs))
        model = Model(bs=bs,m=m, net=args.net, device=device)
        model.train(noisy_imgs_1, noisy_imgs_2, n_epochs)
        out = model.predict(noisy_imgs)
        psnr_ = compute_psnr(out.cpu().float()/256, clean_imgs.float()/256)
        psnr_m.append(psnr_)
        print("m:{}, bs:{}, psnr:{}".format(m,bs,psnr_))
    psnr.append(psnr_m)

torch.save(file_name+'.pickle',{'psnr': psnr, 'bs':bs, 'm':m})


