import torch
import torch.nn as nn
from src.baselines.networks.tcn import *

class UserGenerator(nn.Module):
    def __init__(self, config):
        super(UserGenerator, self).__init__()
        self.n_vars = config.n_vars
        self.noise_dim = config.noise_dim
        self.generators = nn.ModuleList([TCNGenerator(config.G_input_dim, config.hidden_dim) for _ in range(config.n_vars)])        
    def forward(self, batch_size: int, n_steps: int, device: str) -> torch.Tensor:                        
        noise = torch.randn(batch_size, self.noise_dim, n_steps).to(device)
        outputs = torch.cat([self.generators[i](noise) for i in range(self.n_vars)], dim=1)                 
        return outputs

class TCNGenerator(nn.Module):
    def __init__(self, G_input_dim, hidden_dim):
        super(TCNGenerator, self).__init__()
        self.tcn = nn.ModuleList([TemporalBlock(G_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Sequential(nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1))
    def forward(self, z):          
        for layer in self.tcn:
            z = layer(z)
        x = self.last(z)
        return x