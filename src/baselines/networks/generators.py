# import torch
# import torch.nn as nn
# from src.baselines.networks.tcn import *

# class UserGenerator(nn.Module):
#     def __init__(self, config):
#         super(UserGenerator, self).__init__()
#         self.n_vars = config.n_vars
#         self.noise_dim = config.noise_dim
#         self.generators = nn.ModuleList([TCNGenerator(config.G_input_dim, config.hidden_dim) for _ in range(config.n_vars)])        
#     def forward(self, batch_size: int, n_steps: int, device: str) -> torch.Tensor:                        
#         noise = torch.randn(batch_size, self.noise_dim, n_steps).to(device)
#         outputs = torch.cat([self.generators[i](noise) for i in range(self.n_vars)], dim=1)                 
#         return outputs

    
# class TCNGenerator(nn.Module):
#     def __init__(self, G_input_dim, hidden_dim, step_size=16, droptout=0.1):
#         super(TCNGenerator, self).__init__()
#         self.step_size = step_size

#         self.tcn = nn.ModuleList([TemporalBlock(G_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0),
#                                  *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8]]])
#         # Transformer Self-Attention
#         self.self_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim,  # Embedding dimension (matches TCN output channel size)
#             num_heads=4,   # Number of attention heads
#             dropout=droptout,
#             batch_first=True
#         )
#         self.bn = nn.BatchNorm1d(hidden_dim)
#         self.last = nn.Sequential(nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1))        
#         self.gelu = nn.GELU()  # GELU 활성화 함수로 정의        
#         self.dropout = nn.Dropout(droptout)

#     def forward(self, z):
#         # TCN 입력: z.shape == (B, N, T)
        
#         # Pass through TCN layers
#         for layer in self.tcn:
#             z = layer(z)
            
#         B, C, T = z.shape

#         # Reshape z for Self-Attention
#         z_attn = z.view(B, C, -1, self.step_size) 
#         z_attn = z_attn.permute(0, 3, 2, 1).contiguous() 
#         z_attn = z_attn.view(-1, T//self.step_size, C) 

#         # Apply Self-Attention
#         attn_output, _ = self.self_attention(z_attn, z_attn, z_attn) 

#         # Reshape attention output back to original grouped structure
#         attn_output = attn_output.view(B, -1,  T//self.step_size, C).permute(0, 3, 1, 2) 

#         # Reconstruct full sequence
#         attn_output = attn_output.reshape(B, C, T)  # [B, N, T]

#         # Residual connection
#         z = z + self.dropout(attn_output)  # Residual Connection        
#         z = self.bn(z)
#         z = self.gelu(z)        
        
#         # Pass through final layers
#         x = self.last(z)  # [B, 1, T]
#         return x


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
    def __init__(self, G_input_dim, hidden_dim, step_size=16, droptout=0.1):
        super(TCNGenerator, self).__init__()
        self.step_size = step_size

        self.tcn = nn.ModuleList([TemporalBlock(G_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8]]])
        # Transformer Self-Attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,  # Embedding dimension (matches TCN output channel size)
            num_heads=4,   # Number of attention heads
            dropout=droptout,
            batch_first=True
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.last = nn.Sequential(nn.Conv1d(hidden_dim, 1, kernel_size=1, stride=1))        
        self.gelu = nn.GELU()  # GELU 활성화 함수로 정의        
        self.dropout = nn.Dropout(droptout)

    def forward(self, z):
        # TCN 입력: z.shape == (B, N, T)
        
        # Pass through TCN layers
        for layer in self.tcn:
            z = layer(z)
            
        B, C, T = z.shape

        # Reshape z for Self-Attention
        z_attn = z.view(B, C, -1, self.step_size) 
        z_attn = z_attn.permute(0, 3, 2, 1).contiguous() 
        
        B_ = B * self.step_size
        S_ = T // self.step_size
        z_attn = z_attn.view(B_, S_, C)
        
        causal_mask = torch.triu(torch.ones(S_, S_, device=z_attn.device), diagonal=1).bool()
        attn_output, _ = self.self_attention(
            z_attn, z_attn, z_attn,
            attn_mask=causal_mask  # 미래를 보지 못하도록
        )

        # Reshape attention output back to original grouped structure
        attn_output = attn_output.view(B, self.step_size, S_, C)  # (B, step_size, S_, C)
        attn_output = attn_output.permute(0, 3, 2, 1).contiguous()  # (B, C, S_, step_size)
        attn_output = attn_output.view(B, C, T)  # (B, C, T)

        # Residual connection
        z = z + self.dropout(attn_output)  # Residual Connection        
        z = self.bn(z)
        z = self.gelu(z)        
        
        # Pass through final layers
        x = self.last(z)  # [B, 1, T]
        return x

