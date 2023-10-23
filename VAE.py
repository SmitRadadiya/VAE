import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class Encoder(nn.Module):

     def __init__(self,in_channels, latent_dim) -> None:
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.ReLU(True)
        )
        self.flattern = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(86528, 128)   # input dim manual.
        self.mu = nn.Linear(128,latent_dim)
        self.sigma = nn.Linear(128,latent_dim)


     def forward(self, x):
        x = self.encoder(x)
        x = self.flattern(x)
        x = self.fc1(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma
    

class Decoder(nn.Module):
    
    def __init__(self, latent_dim) -> None:
        super(Decoder, self).__init__()

        self.dfc1 = nn.Linear(latent_dim, 128)
        self.dfc2 = nn.Linear(128, 86528)  # check size of image

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 13, 13))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 5, 2, output_padding=1)
        )


    def forward(self, x):
        x = self.dfc1(x)
        x = self.dfc2(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x
    

class VAE(nn.Module):

    def __init__(self,in_channels, latent_dim) -> None:
        super(VAE, self).__init__()

        self.encoder = Encoder(in_channels, latent_dim)

        self.decoder = Decoder(latent_dim)
        self.kl = 0
        self.z = 0


    def reparameterize(self, mu, sigma):
        std = sigma.mul(0.5).exp_()
        eps = torch.randn(mu.shape).to(device)
        return mu + eps * std
    

    def forward(self, x):
        mu, sigma = self.encoder(x)
        self.z = self.reparameterize(mu, sigma)
        recon_x = self.decoder(self.z)

        return recon_x, mu, sigma