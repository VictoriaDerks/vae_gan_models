"""
Utils that support CustomVAE and CustomGAN classes.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def show_img(img, title: str = "", fig_size: tuple = (8, 8)):
    """
    Display a tensor as an image
    :param img: the torch.Tensor object
    :param title: title of the image
    :param fig_size: size of the image
    :return:
    """
    npimg = img.cpu().numpy()

    plt.figure(figsize=fig_size)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def get_gpu(gpu: bool = True):
    """
    Get GPU or CPU object depending on gpu flag
    :param gpu: initialise device as GPU if True, otherwise CPU
    :return:
    """
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f'Using {device}')

    return device


def weights_init(m):
    """
    Custom weights initialization for discriminator and generator.
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def loss_function_vae(recon_x, x, mu, logvar, beta):
    """
    Custom loss function for VAE
    :param recon_x: reconstructed samples
    :param x: original samples
    :param mu:
    :param logvar:
    :param beta:
    :return:
    """
    recon_x = recon_x.reshape(recon_x.shape[0], -1)
    x = x.reshape(x.shape[0], -1)

    # binary cross-entropy loss
    # could use "reduction="mean" too
    b_cross_entropy = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return b_cross_entropy, kld, beta  # beta is returned unchanged


def loss_function_gan(out, label):
    """
    Loss function for GAN, binary crossentropy loss.
    :param out:
    :param label:
    :return:
    """
    criterion = nn.BCELoss(reduction='mean')
    loss = criterion(out, label)
    return loss
