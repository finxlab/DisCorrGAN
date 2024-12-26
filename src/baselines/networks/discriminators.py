import torch
from torch import nn
from torch.nn.utils import spectral_norm
from src.baselines.networks.tcn import TemporalBlock

class UserDiscriminator(nn.Module):
    def __init__(self, config):
        super(UserDiscriminator, self).__init__()
        self.config = config          
        self.discriminators = nn.ModuleList([TCNDiscriminator(config.D_input_dim, config.hidden_dim, config.n_steps) for _ in range(config.n_vars)])        

    def forward(self, x, i):                       
        return self.discriminators[i](x)

class TCNDiscriminator(nn.Module):
    def __init__(self, D_input_dim, hidden_dim, n_steps):
        super(TCNDiscriminator, self).__init__()                    
        self.tcn = nn.ModuleList([TemporalBlock(D_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, spec_norm=True),
                                 *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, spec_norm=True) for i in [1, 2, 4, 8, 16, 32]]])                
        self.last = spectral_norm(nn.Conv1d(hidden_dim, 1, kernel_size=1, dilation=1))
        self.output = nn.Sequential(nn.Linear(n_steps, 1))            

    def forward(self, x): 
        x = torch.cat([x], dim=1) 
        for layer in self.tcn:
            x = layer(x)
        x = self.last(x)
        return self.output(x.squeeze()).reshape(-1, 1)

