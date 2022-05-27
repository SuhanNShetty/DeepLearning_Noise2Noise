import torch
from .model import Model

def compute_psnr(denoised, ground_truth):
    mse = torch.mean((denoised-ground_truth)**2)
    return -10 * torch.log10(mse + 10**(-8))


path_train = '../data/train_data.pkl'
path_val = '../data/val_data.pkl'

if __name__ == "__main__" :
    mod = Model()

    torch.cuda.empty_cache()
    
    noisy_imgs_1, noisy_imgs_2 = torch.load(path_train)
    noisy_imgs, clean_imgs = torch.load(path_val)

    mod.train(noisy_imgs_1, noisy_imgs_2, 100)

    out = mod.predict(noisy_imgs)

    psnr = compute_psnr(out.cpu().float()/256, clean_imgs.float()/256)