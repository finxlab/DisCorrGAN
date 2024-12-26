from torch.nn.functional import one_hot
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset
from src.preprocess.gaussianize import *
from sklearn.preprocessing import StandardScaler

import ml_collections
import yaml
import random


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def set_seed(seed: int):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    torch.manual_seed(seed)
    np.random.seed(seed)

def loader_to_tensor(dl):
    tensor = []
    for x in dl:
        tensor.append(x[0])
    return torch.cat(tensor)


def load_config(file_dir: str):
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config
        

""" Compute the gradient penalty for WGAN-GP for conditional GANs with time series data """
def compute_gradient_penalty(discriminator, real_samples, fake_samples, i):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device).expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    critic_interpolates = discriminator(interpolates, i)
        
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients_norm = gradients.norm(2, dim=(1, 2))
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty

""" Scale the log returns using Gaussianize and StandardScaler """
def scaling(data):
    scalers = [] 
    scaled_data = []

    for i in range(data.shape[1]):  # Iterate over features
        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        gaussianize = Gaussianize()

        # Scale data for the current feature
        feature_data = data[:, i].reshape(-1, 1)  # Reshape to (samples, 1)
        feature_scaled = standardScaler2.fit_transform(
            gaussianize.fit_transform(
                standardScaler1.fit_transform(feature_data)
            )
        )
        # Append scalers and scaled data
        scalers.append((standardScaler1, standardScaler2, gaussianize))
        scaled_data.append(feature_scaled.flatten())  # Flatten back to (samples,)

    # Combine scaled features back into a single array
    scaled_data = np.array(scaled_data).T  # Transpose to shape (samples, features)
    return scaled_data, scalers


""" Inverse the scaling process for all features """
def inverse_scaling(y, scalers):
    y = y.cpu().detach().numpy()  # Convert to NumPy for compatibility with scalers    
    y_original = np.zeros_like(y)  # Placeholder for original data

    for idx in range(y.shape[1]):
        
        standardScaler1, standardScaler2, gaussianize = scalers[idx]
        
        y_feature = y[:, idx, :]  # Shape: (batch_size, seq_len)

        # Normalize by batch mean and std for the current feature
        EPS = 1e-8
        y_feature = (y_feature - y_feature.mean(axis=0, keepdims=True)) / (y_feature.std(axis=0, keepdims=True) + EPS)

        # Perform inverse scaling step-by-step
        y_feature = standardScaler2.inverse_transform(y_feature)
        y_feature = np.array([
            gaussianize.inverse_transform(np.expand_dims(sample, 1)) for sample in y_feature
        ]).squeeze()
        y_feature = standardScaler1.inverse_transform(y_feature)

        # Assign back the feature's inverse transformed data
        y_original[:, idx, :] = y_feature

    return y_original

def correlation_loss(fake: torch.Tensor, real: torch.Tensor, corr_loss_type: str) -> torch.Tensor:

    # Compute correlation matrices
    fake_corr = torch.corrcoef(fake.transpose(1, 2).reshape(-1, fake.shape[1]).T)
    real_corr = torch.corrcoef(real.transpose(1, 2).reshape(-1, real.shape[1]).T)

    # Compute loss
    if corr_loss_type == "l2":
        loss = torch.mean((fake_corr - real_corr) ** 2)
    elif corr_loss_type == "l1":    
        loss = torch.mean(torch.abs(fake_corr - real_corr))
    elif corr_loss_type == "fro":
        loss = torch.norm(fake_corr - real_corr, p="fro") 
    return loss, fake_corr


