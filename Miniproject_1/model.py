class Model():
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function + any other stuff you need
        pass

    def load_pretrained_model(self ) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        pass
        
    def train ( self , train ̇input , train ̇target ) -> None:
        # : train_input : tensor of size (N , C , H , W ) containing a noisy version of the images
        # : train_target : tensor of size (N , C , H , W ) containing another noisy version of the same images, which only differs from the input by their noise .
        pass
    
    def predict ( self , test ̇input ) -> torch.Tensor:
        # :test_input: tensor of size ( N1 , C , H , W ) that has to be denoised by the trained or the loaded network .
        # : returns a tensor of the size ( N1 , C , H , W )
        pass