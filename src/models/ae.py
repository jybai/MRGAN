import torch
from torch import nn
from torch.nn import functional as F

class AE(nn.Module):

    def __init__(self, in_channels, latent_dim, hidden_dims=None, img_size=64, scale=False, **kwargs):
        super(AE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.scale = scale

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.bottleneck_img_size = self.img_size // (2**len(hidden_dims))

        self.encoder = nn.Sequential(
                *modules, 
                nn.Flatten(start_dim=1),
                nn.Linear(hidden_dims[-1]* (self.bottleneck_img_size**2), latent_dim))

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * (self.bottleneck_img_size**2))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution

        return result

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.bottleneck_img_size, self.bottleneck_img_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input, **kwargs):
        if self.scale:
            input = input * 2. - 1. # [0, 1] -> [-1, 1]
        z = self.encode(input)
        return z
    
        '''
        recon_x = self.decode(z)
        
        if 'return_z' in kwargs and kwargs['return_z']:
            return [recon_x, z, input]
        else:
            return [recon_x, input]
        '''

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss}
