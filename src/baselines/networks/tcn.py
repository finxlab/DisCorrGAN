import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm     

"""
This code is partially adapted from the repository:
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
by Shaojie Bai, J. Zico Kolter, and Vladlen Koltun.
"""

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, spec_norm=False, dropout=0.1):
        super(TemporalBlock, self).__init__()

        # First convolutional block
        conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv1 = spectral_norm(conv1) if spec_norm else conv1
        self.chomp1 = None if padding == 0 else Chomp1d(padding)  # Apply Chomp1d only if padding > 0        
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)        
        self.bn1 = nn.BatchNorm1d(n_outputs)   
        
        # Second convolutional block
        conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = spectral_norm(conv2) if spec_norm else conv2
        self.chomp2 = None if padding == 0 else Chomp1d(padding)  # Apply Chomp1d only if padding > 0        
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        # Residual connection
        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = nn.Sequential(
                nn.Conv1d(n_inputs, n_outputs, kernel_size=1),                
                nn.BatchNorm1d(n_outputs)
            )                                
        self.relu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample[0].weight, nonlinearity='leaky_relu')    

    def forward(self, x):
                
        # First convolution block
        out = self.conv1(x)
        if self.chomp1:
            out = self.chomp1(out)                
        out = self.bn1(out)  
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second convolution block
        out = self.conv2(out)
        if self.chomp2:
            out = self.chomp2(out)
        out = self.bn2(out)  
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)        
