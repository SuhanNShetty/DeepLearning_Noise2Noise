import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch import nn
from tqdm import tqdm


class Model():
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need

        # Check the device to use
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the number of in, middle and out channels
        in_channels = 3
        m = 10
        out_channels = 3

        # Initialize the model
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, m, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(m, m, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(m, m, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(m, m, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m, m, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(m * 2, m * 2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(m*2, m*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m*2, m*2, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(m*3, m*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(m*2, m*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(m*2, m*2, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(m*2 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


        pass



    def load_pretrained_model(self ) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        pass
        
    def train ( self , train_input , train_target ) -> None:
        # : train_input : tensor of size (N , C , H , W ) containing a noisy version of the images
        # : train_target : tensor of size (N , C , H , W ) containing another noisy version of the same images, which only differs from the input by their noise .
        pass
    
    def predict ( self , test_input ) -> torch.Tensor:
        # :test_input: tensor of size ( N1 , C , H , W ) that has to be denoised by the trained or the loaded network .
        # : returns a tensor of the size ( N1 , C , H , W )
        pass