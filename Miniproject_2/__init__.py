from .model import Model
import torch

path_train = '../data/train_data.pkl'
path_val = '../data/val_data.pkl'

def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y)**2).mean((1,2,3))).mean()

if __name__ == "__main__" :
    
    model = Model(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    noisy_imgs_1, noisy_imgs_2 = torch.load(path_train)
    noisy_imgs_1 = noisy_imgs_1
    noisy_imgs_2 = noisy_imgs_2
    noisy_imgs , clean_imgs = torch.load(path_val)
    model.optimizer.lr = 1e-2
    
    model.train(noisy_imgs_1, noisy_imgs_2, 500)