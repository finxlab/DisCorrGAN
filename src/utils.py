from torch.nn.functional import one_hot
import torch, math
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


def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """
    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0

def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(1/length, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def AddTime(x):
    """
    Time augmentation for paths
    Parameters
    ----------
    x: torch.tensor, [B, L, D]

    Returns
    -------
    Time-augmented paths, torch.tensor, [B, L, D+1]
    """
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)


def load_config(file_dir: str):
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config
        

""" Compute the gradient penalty for WGAN-GP for conditional GANs with time series data """
# def compute_gradient_penalty(discriminator, real_samples, fake_samples, i):
#     alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device).expand_as(real_samples)
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
#     critic_interpolates = discriminator(interpolates, i)
        
#     gradients = torch.autograd.grad(
#         outputs=critic_interpolates,
#         inputs=interpolates,
#         grad_outputs=torch.ones_like(critic_interpolates),
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
    
#     gradients_norm = gradients.norm(2, dim=(1, 2))
#     gradient_penalty = ((gradients_norm - 1) ** 2).mean()
#     return gradient_penalty
def compute_gradient_penalty(discriminator, real_samples, fake_samples, i):
    """
    Gradient Penalty를 올바르게 계산하는 함수 (WGAN-GP 방식).
    
    Parameters:
        - discriminator: Discriminator 네트워크
        - real_samples: 실제 데이터 (B, 1, T)
        - fake_samples: 생성된 데이터 (B, 1, T)
        - i: Discriminator에 전달될 추가 정보
    
    Returns:
        - gradient_penalty: Gradient Penalty 스칼라 값 (Tensor)
    """
    
    B, C, T = real_samples.shape  # (B, 1, T)
    
    # (1) Alpha 생성 (모든 시점(T)에 대해 동일한 interpolation 적용)
    alpha = torch.rand(B, 1, 1, device=real_samples.device)  # (B, 1, 1)
    
    # (2) Interpolation 계산
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)  # (B, 1, T)

    # (3) Discriminator 평가
    critic_interpolates = discriminator(interpolates, i)  # Shape: (B, 1) 또는 (B, 1, T)

    # (4) Gradient 계산
    grad_outputs = torch.ones_like(critic_interpolates, requires_grad=False)  # Shape 일치 필수
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # Shape: (B, 1, T)

    # (5) Gradient Norm 계산 (Batch-wise)
    gradients_norm = gradients.view(B, -1).norm(2, dim=1)  # (B,)

    # (6) Gradient Penalty 계산
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()  # 스칼라 값

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

def inverse_scaling_split(y, scalers, idx):
    y = y.cpu().detach().numpy()  # Convert to NumPy for compatibility with scalers    
    y_original = np.zeros_like(y)  # Placeholder for original data
        
    standardScaler1, standardScaler2, gaussianize = scalers[idx]
    
    y_feature = y[:, 0, :]  # Shape: (batch_size, seq_len)

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
    y_original[:, 0, :] = y_feature

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



class PositionalEncoding(nn.Module):
    """
    (batch_first=True) 형태를 가정합니다.
    입력: (B, seq_len, d_model)
    출력: (B, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        # [max_len, d_model] 크기의 행렬에 sin/cos 값을 미리 계산해 저장
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스: cos

        # 학습되지 않는 버퍼로 등록 (forward 시 그대로 사용됨)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape: (B, seq_len, d_model)
        seq_len 구간만큼 Positional Encoding을 slice하여 x에 더해준다.
        """
        bsz, seq_len, _ = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"입력 길이({seq_len})가 최대 길이({self.max_len})보다 깁니다.")
        
        # PE를 seq_len만큼 잘라서 (1, seq_len, d_model) 형태로
        pe_slice = self.pe[:, :seq_len, :]  
        x = x + pe_slice  # broadcast되어 (B, seq_len, d_model)에 더해짐
        return x
    