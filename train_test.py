"""
Training and testing functions for VAE
"""

import torch

from utils import loss_function_vae

from torch.utils.data import DataLoader


def train_vae(model, epochs: int, optimiser, loader_train: DataLoader, loader_test: DataLoader, beta: float,
              device: torch.device, verbose: bool = True, save: bool = True):
    """
    Train & evaluate the VAE model
    :param model: untrained model
    :param epochs: number of epochs to train for
    :param optimiser: initialised Adam optimiser
    :param loader_test: testing data as Dataloader object
    :param loader_train: training data as Dataloader object
    :param beta: beta to use when calculating loss
    :param device: to sent tensors to, either a GPU or CPU
    :param save: save trained model to disk
    :param verbose: print loss every 100 iterations
    :return: trained model and lists with train/test losses
    """
    # loss lists init
    # (train)
    total_train_loss = []
    reconstruction_train_loss = []
    kl_train_loss = []

    # (test)
    total_test_loss = []
    reconstruction_test_loss = []
    kl_test_loss = []

    model.train()  # set mode to training

    for epoch in range(epochs):
        data = None
        train_loss = 0
        train_kl_loss = 0
        train_rl_loss = 0

        for batch_idx, data in enumerate(loader_train):

            data, _ = data
            data = data.to(device)

            optimiser.zero_grad()

            reconstr_batch, mu, logvar = model(data)  # forward pass

            rl, kld, beta = loss_function_vae(reconstr_batch, data, mu, logvar, beta)

            loss = rl + beta * kld

            loss.backward()

            train_loss += loss.item()
            train_kl_loss += kld.item()
            train_rl_loss += rl.item()

            optimiser.step()

            if verbose:
                if batch_idx % 100 == 0:  # print loss every so often
                    print(f'Epoch: {epoch}, Iteration {batch_idx}, loss = {round(loss.item() / len(data), 4)}')
                    print()

        # get average loss for the epoch
        epoch_total_train_loss = train_loss / len(loader_train.dataset)
        epoch_kl_train_loss = train_kl_loss / len(loader_train.dataset)
        epoch_rl_train_loss = train_rl_loss / len(loader_train.dataset)

        # evaluate model on test set at end of epoch
        epoch_total_test_loss, epoch_kl_test_loss, epoch_rl_test_loss = evaluate_vae(beta=beta,
                                                                                     model=model,
                                                                                     loader_test=loader_test,
                                                                                     device=device)

        # save test losses
        total_test_loss.append(epoch_total_test_loss)
        reconstruction_test_loss.append(epoch_kl_test_loss)
        kl_test_loss.append(epoch_rl_test_loss)

        # save train losses
        total_train_loss.append(epoch_total_train_loss)
        reconstruction_train_loss.append(epoch_kl_train_loss)
        kl_train_loss.append(epoch_rl_train_loss)

        # save the final model to disk
        if save:
            if epoch == epochs - 1:
                with torch.no_grad():
                    torch.jit.save(torch.jit.trace(model, (data), check_trace=False),
                                   'saved_models/VAE_model.pth')

    return model, total_train_loss, reconstruction_train_loss, kl_train_loss, \
           total_test_loss, reconstruction_test_loss, kl_test_loss


def evaluate_vae(beta, model, loader_test, device):
    """
    Test the model, called by train_vae()
    :param model: trained VAE model
    :param beta: beta to use when calculating loss
    :param loader_test: test set as DataLoader object
    :param device: to sent tensors to, either a GPU or CPU
    :return:
    """
    model.eval()  # switch to evaluation mode

    test_loss = 0
    test_kl_loss = 0
    test_rl_loss = 0

    for i, data in enumerate(loader_test):
        data, _ = data
        data = data.to(device)

        reconstr_batch, mu, logvar = model(data)

        test_rl, test_kld, test_beta = loss_function_vae(reconstr_batch, data, mu, logvar, beta)

        loss = test_rl + test_beta * test_kld

        test_kl_loss += test_kld.item()
        test_rl_loss += test_rl.item()
        test_loss += loss.item()

    test_loss /= len(loader_test.dataset)
    test_kl_loss /= len(loader_test.dataset)
    test_rl_loss /= len(loader_test.dataset)

    print(f"====> Test set loss: {test_loss}")

    return test_loss, test_kl_loss, test_rl_loss
