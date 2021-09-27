"""
Implementation of a VAE model.
- Training, testing
- testing the effect of different betas
- visualising generated and reconstructed inputs
- looking at latent representations with TSNE
"""

import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models import VAE
from train_test import train_vae
from utils import get_gpu, show_img

from sklearn.manifold import TSNE
# importing seaborn for pretty plots
import seaborn as sns
import pandas as pd  # for dataframe

# random seed for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)


class CustomVAE:
    def __init__(self, gpu_flag: bool, epochs: int, lr: float, batch: int, latent: int, verbose: bool = False):
        # transforms for input
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch
        self.latent_dim = latent

        # get data loaders
        self.loader_train, self.loader_test = self.load_data()

        self.device = get_gpu(gpu_flag)

        # initialised model and optimiser (Adam)
        self.model, self.optimiser = self.get_model(verbose=verbose)

    @staticmethod
    def denorm(x):
        return x

    def load_data(self):
        """
        Load MNIST data and put them into train and test dataloaders.
        :return: dataloader objects (train, test)
        """
        training_data = datasets.MNIST("data/", train=True, download=True, transform=self.transform)

        testing_data = datasets.MNIST("data/", train=False, transform=self.transform)

        loader_train = DataLoader(training_data, self.batch_size, shuffle=True)
        loader_test = DataLoader(testing_data, self.batch_size, shuffle=False)  # no need to shuffle

        return loader_train, loader_test

    def get_model(self, verbose: bool):
        """
        Initialise model object and optimiser
        :return: model object and optimiser object (Adam)
        """
        model = VAE(self.latent_dim).to(self.device)

        if verbose:
            params = sum(param.numel() for param in model.parameters() if param.requires_grad)
            print("Total number of parameters is: {}".format(params))
            print(model)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        return model, optimizer

    def train_test(self, beta: float, epochs=None, verbose: bool = False, save: bool = True):
        """
        Calls training function.
        :param beta:
        :param epochs: (Optional) specify epochs, otherwise epoch number from class intialisation is used.
        :param verbose:
        :param save: save down the trained model, default = True
        :return:
        """
        if epochs is None:
            epochs = self.epochs

        output = train_vae(model=self.model, epochs=epochs, optimiser=self.optimiser, device=self.device,
                           loader_train=self.loader_train, loader_test=self.loader_test, beta=beta,
                           verbose=verbose, save=save)

        self.model = output[0]  # save trained model

        return output[1:]

    def test_betas(self, epochs=None, beta_list=None):
        """
        Test the effects of specifying different betas.
        :param epochs: (Optional) specify epochs, otherwise epoch number from class intialisation is used.
        :param beta_list: (Optional) betas to test. Otherwise, betas tested are 0.1, 0.5, 1, 2, 2.5, 3, 4, 5
        :return:
        """
        if beta_list is None:
            beta_list = [0.1, 0.5, 1, 2, 2.5, 3, 4, 5]

        diff_betas = {}  # dictionary saving train, test loss for different betas

        # test with different betas
        for beta_num in beta_list:
            (total_train_loss, reconstruction_train_loss, KL_train_loss,
             total_test_loss, reconstruction_test_loss, KL_test_loss) = self.train_test(beta_num, epochs=epochs,
                                                                                        verbose=False)

            average_train_loss = torch.mean(torch.FloatTensor(total_train_loss))
            average_test_loss = torch.mean(torch.FloatTensor(total_test_loss))
            diff_betas[beta_num] = (average_test_loss, average_train_loss)

            print(f"For beta {beta_num}, average test loss across {self.epochs} epochs is {average_test_loss} "
                  f"and average train loss is {average_train_loss}.")

            to_plot = [(total_test_loss, total_train_loss, "Total"),
                       (reconstruction_test_loss, reconstruction_train_loss, "Reconstruction"),
                       (KL_test_loss, KL_train_loss, "KL")]

            plt.style.use("dark_background")

            # plot losses for different betas
            for dataset in to_plot:
                self.create_plot(dataset, beta_num, epochs)

    def create_plot(self, data_in, beta_val: float, epochs: int = None):
        """
        Create plot of train and test losses.
        :param beta_val:
        :param data_in:
        :param epochs:
        :return:
        """
        if epochs is None:
            epochs = self.epochs

        test = data_in[0]
        train = data_in[1]
        label = data_in[2]
        total_test_plt = plt.plot(np.arange(1, epochs + 1, 1), test, label="test")
        total_train_plt = plt.plot(np.arange(1, epochs + 1, 1), train, label="train")
        plt.title(f"{label} loss for train and test, beta = {beta_val}")

        plt.legend()
        plt.ylabel(f"{label} loss")
        plt.xlabel("epochs")

        plt.show()

    def compare_reconstructions(self):
        """
        Function to complete the input images, their reconstructions, and generated images.
        :return:
        """
        # get inputs from test set
        sample_inputs, _ = next(iter(self.loader_test))
        fixed_input = sample_inputs[0:32, :, :, :]

        # visualize the original images of the last batch of the test set
        img = make_grid(self.denorm(fixed_input), nrow=8, padding=2, normalize=False,
                        range=None, scale_each=False, pad_value=0)

        show_img(img, title="Input images")

        with torch.no_grad():
            # visualize the reconstructed images of the last batch of test set

            fixed_input = fixed_input.to(self.device)
            recon_batch, _, _ = self.model(fixed_input)  # run input images through model

            recon_batch = recon_batch.cpu()
            recon_batch = make_grid(self.denorm(recon_batch), nrow=8, padding=2, normalize=False,
                                    range=None, scale_each=False, pad_value=0)

            show_img(recon_batch, title='Reconstructed images')

        self.model.eval()
        n_samples = 256
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        with torch.no_grad():
            # visualise the generated images

            z = z.to(self.device)
            samples = self.model.decode(z)  # decode randomly-generated numbers into an image

            samples = samples.cpu()
            samples = make_grid(self.denorm(samples), nrow=16, padding=2, normalize=False,
                                range=None, scale_each=False, pad_value=0)

            show_img(samples, title='Generated Images')

    def get_test_rep(self):
        """
        Get latent representation of test set data
        :return:
        """
        with torch.no_grad():
            test_data = (self.loader_test.dataset.data).float()  # need to convert to float
            test_data /= 255  # effect of .float() is range of 0-255, bring it back to range of 0-1
            test_data = test_data.view(-1, 1, 28, 28)  # reshape
            # print(test_data.shape)

            test_data = test_data.to(self.device)  # send to gpu for fast testing

            mu, logvar = self.model.encode(test_data)
            test_rep = self.model.reparametrize(mu, logvar)
            return test_rep

    def visualise_tsne(self):
        """
        TSNE visualisation for latent representations
        :return:
        """
        test_rep = self.get_test_rep().cpu()  # send to cpu because of TSNE not working on GPU
        z_embedded = TSNE(n_components=2).fit_transform(test_rep)

        sns.set_style("dark")  # because my IDE is in dark mode
        plt.style.use("dark_background")

        test_labels = (self.loader_test.dataset.targets)

        # unique_labels = set(test_labels.numpy())
        # print(unique_labels)

        data = pd.DataFrame(z_embedded, columns=["first_dim", "second_dim"])
        data["label"] = test_labels

        # data.head()

        plt.figure(figsize=(10, 10))

        sns.scatterplot(x='first_dim',
                        y='second_dim',
                        hue="label",
                        data=data,
                        palette=sns.color_palette("deep"),
                        linewidth=0.1,  # no borders on dots, looks prettier
                        legend="full",
                        alpha=0.7)

        plt.show()


if __name__ == '__main__':
    GPU = True  # Choose whether to use GPU     

    # Necessary Hyperparameters
    num_epochs = 20  # increasing this would give a better trained model
    learning_rate = 0.0001
    batch_size = 64
    latent_dim = 20  # seems to be enough for a dataset like MNIST which isn't complex

    verbose_flag = True
    save_model = False

    vae = CustomVAE(gpu_flag=GPU, epochs=num_epochs, lr=learning_rate, batch=batch_size, latent=latent_dim,
                    verbose=verbose_flag)

    # train model
    vae.train_test(beta=2, epochs=20, verbose=verbose_flag, save=save_model)

    # test different beta values
    vae.test_betas(epochs=5, beta_list=[0.5, 2, 5])

    # compare original samples, against reconstructed samples, against generated samples
    vae.compare_reconstructions()

    # show plot of latent representations analysed with TSNE
    vae.visualise_tsne()
