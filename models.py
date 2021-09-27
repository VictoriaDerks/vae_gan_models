"""
GAN and VAE model classes, as well as supporting classes.
"""


import torch.nn as nn
import torch

class Flatten(nn.Module):
    """
    Flattens data into a 2D vector to feed to dense bottleneck in VAE.
    Operationalised as a class so it can be included in the model definition.
    """
    def forward(self, input_array):
        return input_array.view(input_array.size(0), -1)


class UnFlatten(nn.Module):
    """
    Un-flattens data into a 4D vector.
    Operationalised as a class so it can be included in the model definition.
    512 is hardcoded, represents the number of input neurons of the VAE decoder.
    """
    def forward(self, input_array):
        # "size" parameter is dependent on the input size of the decoder
        return input_array.view(input_array.size(0), 512, 1, 1)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            Flatten()  # for easy processing in the dense layers
        )

        # linear layers in the middle, "dense bottleneck" starts here

        # from flattened encoder output -> big input hidden layer
        self.fc1 = nn.Linear(512, 1024)

        # from big input hidden layer -> latent dimensions
        self.fc21 = nn.Linear(1024, latent_dim)  # mu
        self.fc22 = nn.Linear(1024, latent_dim)  # logvar

        # latent dimenions ->  back to big hidden layer
        self.fc3 = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU()
        )

        # big hidden layer -> input size for the decoder
        self.fc4 = nn.Linear(1024, 512)

        # decoder
        self.decoder = nn.Sequential(
            UnFlatten(),  # reshape the output of the dense layers
            nn.ConvTranspose2d(in_channels=512, out_channels=64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2),
            nn.Sigmoid()  # sigmoid to get values between 0 - 1 (otherwise loss doesn't work)
        )

    def encode(self, x):
        encoded = self.encoder(x)
        encoded = self.fc1(encoded)

        mu = self.fc21(encoded)
        logvar = self.fc22(encoded)

        return mu, logvar

    def reparametrize(self, mu, logvar):
        """
        Reparametrization trick, https://arxiv.org/pdf/1312.6114v10.pdf section 2.4 and 3.
        :param mu:
        :param logvar:
        :return:
        """
        stdev = logvar.mul(0.5).exp_()  # multiply tensor with 0.5, then take the exponential of the tensor
        noise = torch.randn_like(stdev)  # random noise
        z = mu + (noise * stdev)
        return z

    def decode(self, z):
        # encoder but reversed
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar


class Generator(nn.Module):
    """
    Generator for the GAN, based on https://arxiv.org/abs/1511.06434
    """
    def __init__(self, latent_vector):
        super(Generator, self).__init__()
        # original DCGAN: 1024 -> 512 -> 256 -> 128 -> 3, https://arxiv.org/abs/1511.06434 figure 1
        # this GAN: 512 -> 256 -> 128 -> 64 -> 3

        # structure of each block:
        # -- convolution layer
        # -- batch norm layer
        # -- relu layer

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_vector, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, label):
        out = self.generator(z)
        return out


class Discriminator(nn.Module):
    """
    Discriminator for the GAN, based on https://arxiv.org/abs/1511.06434
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        # this GAN: 3 -> 64 -> 128 -> 265 -> 512

        # structure of each block:
        # -- convolution layer
        # -- batch norm layer (except first & last block)
        # -- leaky relu layer

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # 0.2 is what's used in the original paper

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()  # final activation is sigmoid
        )

    def forward(self, x, label):
        out = self.discriminator(x)
        out = out.view(-1, 1).squeeze(1)  # flatten

        return out
