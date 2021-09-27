"""
Implementation of a GAN model based on DCGAN.
- Training, testing of discriminator and generator
- visualising generator and discriminator loss
- comparing generated images and dataset images
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

import models
from utils import show_img, get_gpu, loss_function_gan, weights_init

# random seed for reproducibility
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class CustomGAN:
    def __init__(self, latent_vec, batch: int, epochs: int, lr: float, gpu_flag: bool = True, verbose: bool = False):
        self.scheduler_d = None
        self.scheduler_g = None

        # transforming input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(32),  # done on imagenet in original paper
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.device = get_gpu(gpu_flag)

        self.latent_vector_size = latent_vec
        self.learning_rate = lr
        self.batch_size = batch
        self.epochs = epochs

        self.fixed_noise = torch.randn(batch, latent_vec, 1, 1, device=self.device)

        # label smoothing
        self.real_label = 0.9
        self.fake_label = 0.1

        # get dataloader for train and test set
        self.loader_train, self.loader_test = self.load_data()

        # get generator and discriminator model
        self.model_g, self.model_d = self.get_model(verbose=verbose)

        # get optimisers and schedulers for generator and discriminator
        self.optimiser_d, self.optimiser_g, scheduler_d, scheduler_g = self.get_optimiser()

    @staticmethod
    def denorm(x, channels=None, w=None, h=None, resize=False):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        if resize:
            if channels is None or w is None or h is None:
                print('Number of channels, width and height must be provided for resize.')
            x = x.view(x.size(0), channels, w, h)
        return x

    def load_data(self):
        """
        Load the CIFAR10 data, apply transformations, and put them into train and test dataloaders.
        (Optional) save the train and test sets to disk
        :return:
        """
        data_dir = 'data/'

        cifar10_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=self.transform)
        cifar10_test = datasets.CIFAR10(data_dir, train=False, transform=self.transform)

        loader_train = DataLoader(cifar10_train, batch_size=self.batch_size)
        loader_test = DataLoader(cifar10_test, batch_size=self.batch_size)

        return loader_train, loader_test

    def get_model(self, use_weights_init: bool = True, verbose: bool = False):
        """
        Initialised generator and discriminator.
        :param use_weights_init: whether to use custom weights initialisation for the layers
        :param verbose:
        :return:
        """
        model_g = models.Generator(self.latent_vector_size).to(self.device)
        if use_weights_init:
            model_g.apply(weights_init)

        model_d = models.Discriminator().to(self.device)
        if use_weights_init:
            model_d.apply(weights_init)

        if verbose:
            params_g = sum(p.numel() for p in model_g.parameters() if p.requires_grad)
            print("Total number of parameters in Generator is: {}".format(params_g))
            print(model_g)
            print('\n')

            params_d = sum(p.numel() for p in model_d.parameters() if p.requires_grad)
            print("Total number of parameters in Discriminator is: {}".format(params_d))
            print(model_d)
            print('\n')

            print("Total number of parameters is: {}".format(params_g + params_d))

        return model_g, model_d

    def get_optimiser(self):
        """
        Get Adam optimiser & schedulers for generator and discriminator.
        Scheduler for the discriminator has a bigger step size than scheduler for generator to improve performance.
        :return:
        """
        optimiser_d = torch.optim.Adam(self.model_d.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        optimiser_g = torch.optim.Adam(self.model_g.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

        # schedulers
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimiser_d, step_size=2, gamma=0.95)
        scheduler_g = torch.optim.lr_scheduler.StepLR(optimiser_g, step_size=1, gamma=0.95)

        return optimiser_d, optimiser_g, scheduler_d, scheduler_g

    def train_gan(self, save_img: bool = False, save_model: bool = False,
                  show_loss: bool = False, show_fakes: bool = True, verbose: bool = False):
        """
        Training the GAN, keeping track of losses.
        model_d = discriminator, model_g = generator
        :return:
        """
        train_losses_list_g = []
        train_losses_list_d = []

        for epoch in range(self.epochs):
            train_loss_g = 0
            train_loss_d = 0
            for i, data in enumerate(self.loader_train, 0):
                # Step 1: Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))

                # First, train discriminator with real data.

                self.model_d.zero_grad()

                target = data[1].to(self.device)
                data = data[0].to(self.device)
                # print("data shape: ", data.shape)
                # print("target shape: ", target.shape)

                # need to get batch size each time, because last batch is only 80 samples
                current_batch_size = data.size(0)

                # create labels of same shape as data
                label = torch.full((current_batch_size,),
                                   self.real_label,  # == 0.9
                                   dtype=torch.float, device=self.device)

                output = self.model_d(data, target)

                loss_real = loss_function_gan(output, label)

                loss_real.backward()

                # Then, generate fake data with generator and train discriminator with fake data

                # generate fakes
                current_noise = torch.randn(current_batch_size, self.latent_vector_size, 1, 1, device=self.device)

                fake = self.model_g(current_noise, target)  # use generator creating fake data

                label.fill_(
                    self.fake_label  # == 0.1
                )

                output = self.model_d(fake.detach(), target)  # train discriminator with fake data

                loss_fake = loss_function_gan(output, label)

                loss_fake.backward()

                # total error
                error_discriminator = loss_real + loss_fake
                train_loss_d += error_discriminator.item()

                self.optimiser_d.step()

                # (2) Update G network: maximize log(D(G(z)))

                self.model_g.zero_grad()
                label.fill_(self.real_label)  # fill with 0.9 again, got filled with 0.1 previously

                output = self.model_d(fake, label)  # put fakes through discriminator w/ real label

                loss_generator = loss_function_gan(output, label)
                train_loss_g += loss_generator.item()

                loss_generator.backward()

                self.optimiser_g.step()

                if verbose:
                    print(f"iteration: {i} / {len(self.loader_train)}, epoch {epoch}/{self.epochs}...")

                    print(f'[{epoch}/{self.epochs}][{i}/{len(self.loader_train)}] '
                          f'Discriminator error: {round(error_discriminator.item(), 4)}, '
                          f'Generator error: {round(loss_generator.item(), 4)} ')

            # run scheduler after each epoch
            if self.scheduler_d is not None:
                self.scheduler_d.step()
            if self.scheduler_g is not None:
                self.scheduler_g.step()

            if save_img:
                if epoch == 0:
                    save_image(self.denorm(data.cpu()).float(), 'img/real_samples.png')

                # save generated samples per epoch to compare their quality
                with torch.no_grad():
                    label = torch.full((self.batch_size,),
                                       self.real_label,  # == 0.9
                                       dtype=torch.float, device=self.device)
                    fake = self.model_g(self.fixed_noise, label)
                    save_image(self.denorm(fake.cpu()).float(),
                               f'img/fake_samples_epoch_{epoch}.png')

            train_losses_list_d.append(train_loss_d / len(self.loader_train))
            train_losses_list_g.append(train_loss_g / len(self.loader_train))

        label = torch.full((128,),
                           self.real_label,  # == 0.9, placeholder label for trace function, not actually used
                           dtype=torch.float,
                           device=self.device)
        if save_model:
            # save models
            torch.jit.save(torch.jit.trace(self.model_g, (self.fixed_noise, label), ),
                           'saved_models/GAN_G_model.pth')
            torch.jit.save(torch.jit.trace(self.model_d, (fake, label), ),
                           'saved_models/GAN_D_model.pth')

        if show_loss:
            self.visualise_losses(train_losses_list_d, train_losses_list_g)

        if show_fakes:
            self.show_generated_samples(label, save=save_img)

        return train_losses_list_d, train_losses_list_g, label

    def show_generated_samples(self, label, save: bool):
        """
        Show generated samples created with generator. Then compare them to some real images.
        :param save: whether to save the generated samples
        :param label: vector filled with 0.9 to indicate we want to generate real samples.
        :return:
        """

        input_noise = torch.randn(100, self.latent_vector_size, 1, 1, device=self.device)

        with torch.no_grad():
            # visualize the generated images
            generated = self.model_g(input_noise, label).cpu()

            generated = make_grid(self.denorm(generated)[:100], nrow=10, padding=2, normalize=False,
                                  range=None, scale_each=False, pad_value=0)

            plt.figure(figsize=(15, 15))

            if save:
                save_image(generated, 'img/final_generated_samples.png')

            show_img(generated, "generated samples")

        # visualize the original images of the last batch of the test set for comparison
        it = iter(self.loader_test)
        sample_inputs, _ = next(it)
        fixed_input = sample_inputs[0:64, :, :, :]

        img = make_grid(self.denorm(fixed_input), nrow=8, padding=2, normalize=False,
                        range=None, scale_each=False, pad_value=0)
        plt.figure(figsize=(15, 15))

        show_img(img, "original samples")

    @staticmethod
    def visualise_losses(losses_d, losses_g):
        """
        Plot generator and discriminator losses.
        :param losses_d:
        :param losses_g:
        :return:
        """
        plt.style.use("dark_background")
        plt.plot(np.arange(0, len(losses_d)), losses_d, label="discriminator")
        plt.plot(np.arange(0, len(losses_g)), losses_g, label="generator")
        plt.legend()

        plt.ylabel("Loss")
        plt.xlabel("epochs")

        plt.title("Generator and discriminator loss over the epochs")

        plt.show()


if __name__ == '__main__':
    GPU = True  # Choose whether to use GPU

    # hyperparameters
    num_epochs = 50
    learning_rate = 0.0002  # same as original DCGAN paper
    latent_vector_size = 100
    batch_size = 128

    verbose_flag = True

    gan = CustomGAN(latent_vec=latent_vector_size, batch=batch_size, lr=learning_rate, epochs=num_epochs,
                    gpu_flag=GPU, verbose=verbose_flag)

    # bools for processes to execute after/during training
    save_img_flag = True
    save_models_flag = True
    visualise_losses_flag = True
    visualise_generated_samples_flag = True

    # train the GAN
    gan.train_gan(save_img=save_img_flag, save_model=save_models_flag,
                  show_loss=visualise_losses_flag, show_fakes=visualise_generated_samples_flag,
                  verbose=verbose_flag)
