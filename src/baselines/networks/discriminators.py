import torch
from torch import nn
from torch.nn.utils import spectral_norm
from src.baselines.networks.tcn import TemporalBlock
from src.utils import PositionalEncoding

class UserDiscriminator(nn.Module):
    def __init__(self, config):
        super(UserDiscriminator, self).__init__()
        self.config = config                  
        self.discriminators = nn.ModuleList([TCNDiscriminator(config) for _ in range(config.n_vars)])        
    def forward(self, x, i):                       
        return self.discriminators[i](x)

class TCNDiscriminator(nn.Module):
    def __init__(self, config, dropout=0.1):
        super(TCNDiscriminator, self).__init__()                    
        self.step_size = config.step_size

        self.tcn = nn.ModuleList([TemporalBlock(1, config.hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, dropout=dropout, spec_norm=True),
                                 *[TemporalBlock(config.hidden_dim, config.hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, dropout=dropout, spec_norm=True) for i in [1, 2, 4, 8]]])                
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
        self.output = nn.Sequential(nn.Linear(config.n_steps, 1))  
        
        self.final_ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),            
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(self, x): 
        for layer in self.tcn:
            x = layer(x)
            
        B, C, T = x.shape
        x = x.permute(0, 2, 1)
        x_attn = self.pos_encoder(x)
        x = x.permute(0, 2, 1)
        
        x_attn = x.view(B, C, -1, self.step_size) 
        x_attn = x_attn.permute(0, 3, 2, 1).contiguous()         
        
        B_ = B * self.step_size
        S_ = T // self.step_size
        x_attn = x_attn.view(B_, S_, C)
        
        # 3) Positional Encoding 주입        
        x_attn = self.pos_encoder(x_attn)

        positions = torch.arange(S_, device=x_attn.device).unsqueeze(1)  # shape: (S_, 1)
        attn_mask = (positions - positions.transpose(0, 1) < 0) | (positions - positions.transpose(0, 1) >= 8)
        attn_output, _ = self.self_attention(
            x_attn, x_attn, x_attn,
            attn_mask=attn_mask  
        )

        attn_output = attn_output.view(B, self.step_size, S_, C) 
        attn_output = attn_output.permute(0, 3, 2, 1).contiguous()
        attn_output = attn_output.view(B, C, T) 
        
        # Residual connection
        x = x + self.dropout(attn_output) 
        x = self.bn1(x)        
        
        x = x.permute(0, 2, 1)  # (B, T, C)
        out = self.final_ffn(x)  # (B, T, 1)
        out = out.permute(0, 2, 1)  # (B, 1, T)
        out = self.bn2(out) 
        return self.output(out.squeeze()).reshape(-1, 1)