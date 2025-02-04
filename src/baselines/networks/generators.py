import torch
import torch.nn as nn
from src.baselines.networks.tcn import *
from src.utils import PositionalEncoding

class UserGenerator(nn.Module):
    def __init__(self, config):
        super(UserGenerator, self).__init__()
        self.n_vars = config.n_vars
        self.noise_dim = config.noise_dim
        self.generators = nn.ModuleList([TCNGenerator(config) for _ in range(config.n_vars)])        
    def forward(self, batch_size: int, n_steps: int, device: str) -> torch.Tensor:                        
        noise = torch.randn(batch_size, self.noise_dim, n_steps).to(device)
        outputs = torch.cat([self.generators[i](noise) for i in range(self.n_vars)], dim=1)                 
        return outputs

class TCNGenerator(nn.Module):
    def __init__(self, config, dropout=0.1):
        super(TCNGenerator, self).__init__()
        self.step_size = config.step_size

        self.tcn = nn.ModuleList([TemporalBlock(config.noise_dim, config.hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, dropout=dropout),
                                 *[TemporalBlock(config.hidden_dim, config.hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i,  dropout=dropout) for i in [1, 2, 4, 8]]])
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim, 
            num_heads=4,  
            dropout=dropout,
            batch_first=True
        )        
        self.pos_encoder = PositionalEncoding(d_model=config.hidden_dim, max_len=1024)        
        self.bn1 = nn.BatchNorm1d(config.hidden_dim)                
        self.bn2 = nn.BatchNorm1d(1)      
        self.dropout = nn.Dropout(dropout)
        
        self.final_ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),        
            nn.GELU(),
            nn.Dropout(dropout),            
            nn.Linear(config.hidden_dim // 2, 1)
        )                

    def forward(self, z):
        for layer in self.tcn:
            z = layer(z)
            
        B, C, T = z.shape

        z = z.permute(0, 2, 1)  
        z_attn = self.pos_encoder(z)
        z = z.permute(0, 2, 1)  
        
        # Reshape z for Self-Attention
        z_attn = z.view(B, C, -1, self.step_size) 
        z_attn = z_attn.permute(0, 3, 2, 1).contiguous() 
        
        B_ = B * self.step_size
        S_ = T // self.step_size
        z_attn = z_attn.view(B_, S_, C)
        
        """ 3) Positional Encoding 추가 """
        z_attn = self.pos_encoder(z_attn)  # (B_, S_, C)

        positions = torch.arange(S_, device=z_attn.device).unsqueeze(1)  # shape: (S_, 1)        
        attn_mask = (positions - positions.transpose(0, 1) < 0) | (positions - positions.transpose(0, 1) >= 8)        
        
        attn_output, _ = self.self_attention(
            z_attn, z_attn, z_attn,
            attn_mask=attn_mask  
        )
        
        attn_output = attn_output.view(B, self.step_size, S_, C) 
        attn_output = attn_output.permute(0, 3, 2, 1).contiguous()
        attn_output = attn_output.view(B, C, T) 

        # Residual connection
        z = z + self.dropout(attn_output)
        z = self.bn1(z) 
        
        z = z.permute(0, 2, 1)  # (B, T, C)
        z = self.final_ffn(z)  # (B, T, 1)
        z = z.permute(0, 2, 1)  # (B, 1, T)
        out = self.bn2(z)  # (B, 1, T)        
        return out