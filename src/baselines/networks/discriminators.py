# import torch
# from torch import nn
# from torch.nn.utils import spectral_norm
# from src.baselines.networks.tcn import TemporalBlock

# class UserDiscriminator(nn.Module):
#     def __init__(self, config):
#         super(UserDiscriminator, self).__init__()
#         self.config = config          
        

#         self.discriminators = nn.ModuleList([TCNDiscriminator(config.D_input_dim, config.hidden_dim, config.n_steps) for _ in range(config.n_vars)])        

#     def forward(self, x, i):                       
#         return self.discriminators[i](x)

# class TCNDiscriminator(nn.Module):
#     def __init__(self, D_input_dim, hidden_dim, n_steps, dropout=0.1, step_size=16):
#         super(TCNDiscriminator, self).__init__()                    
#         self.step_size = step_size

#         self.tcn = nn.ModuleList([TemporalBlock(D_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, spec_norm=True),
#                                  *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, spec_norm=True) for i in [1, 2, 4, 8]]])                
#         self.self_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,  # Embedding dimension (matches TCN output channel size)
#             num_heads=4,   # Number of attention heads
#             dropout=dropout,
#             batch_first=True
#         )
#         self.bn = nn.BatchNorm1d(hidden_dim)
#         self.last = spectral_norm(nn.Conv1d(hidden_dim, 1, kernel_size=1, dilation=1))
#         self.output = nn.Sequential(nn.Linear(n_steps, 1))            
        
#         self.gelu = nn.GELU()  # GELU 활성화 함수로 정의

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x): 
#         for layer in self.tcn:
#             x = layer(x)
            
#         B, C, T = x.shape

#         # Reshape z for Self-Attention
#         x_attn = x.view(B, C, -1, self.step_size) 
#         x_attn = x_attn.permute(0, 3, 2, 1).contiguous() 
#         x_attn = x_attn.view(-1, T//self.step_size, C) 

#         # Apply Self-Attention
#         attn_output, _ = self.self_attention(x_attn, x_attn, x_attn) 

#         # Reshape attention output back to original grouped structure
#         attn_output = attn_output.view(B, -1,  T//self.step_size, C).permute(0, 3, 1, 2) 

#         # Reconstruct full sequence
#         attn_output = attn_output.reshape(B, C, T)  # [B, N, T]
        
#         # Residual connection
#         x = x + self.dropout(attn_output)  # Residual Connection        
#         x = self.bn(x)
#         #x = self.relu(x)
#         x = self.gelu(x)
        
#         x = self.last(x)
#         return self.output(x.squeeze()).reshape(-1, 1)

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
    def __init__(self, D_input_dim, hidden_dim, n_steps, dropout=0.1, step_size=16):
        super(TCNDiscriminator, self).__init__()                    
        self.step_size = step_size

        self.tcn = nn.ModuleList([TemporalBlock(D_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, spec_norm=True),
                                 *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, spec_norm=True) for i in [1, 2, 4, 8]]])                
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,  # Embedding dimension (matches TCN output channel size)
            num_heads=4,   # Number of attention heads
            dropout=dropout,
            batch_first=True
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.last = spectral_norm(nn.Conv1d(hidden_dim, 1, kernel_size=1, dilation=1))
        self.output = nn.Sequential(nn.Linear(n_steps, 1))            
        
        self.gelu = nn.GELU()  # GELU 활성화 함수로 정의

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        for layer in self.tcn:
            x = layer(x)
            
        B, C, T = x.shape

        # Reshape z for Self-Attention
        x_attn = x.view(B, C, -1, self.step_size) 
        x_attn = x_attn.permute(0, 3, 2, 1).contiguous() 
        
        
        B_ = B * self.step_size
        S_ = T // self.step_size
        x_attn = x_attn.view(B_, S_, C)

        causal_mask = torch.triu(torch.ones(S_, S_, device=x_attn.device), diagonal=1).bool()
        attn_output, _ = self.self_attention(
            x_attn, x_attn, x_attn,
            attn_mask=causal_mask  # 미래를 보지 못하도록
        )

        # Reshape attention output back to original grouped structure
        attn_output = attn_output.view(B, self.step_size, S_, C)  # (B, step_size, S_, C)
        attn_output = attn_output.permute(0, 3, 2, 1).contiguous()  # (B, C, S_, step_size)
        attn_output = attn_output.view(B, C, T)  # (B, C, T)
        
        # Residual connection
        x = x + self.dropout(attn_output)  # Residual Connection        
        x = self.bn(x)
        #x = self.relu(x)
        x = self.gelu(x)
        
        x = self.last(x)
        return self.output(x.squeeze()).reshape(-1, 1)