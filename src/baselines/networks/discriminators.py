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

# class TCNDiscriminator(nn.Module):
#     def __init__(self, D_input_dim, hidden_dim, n_steps):
#         super(TCNDiscriminator, self).__init__()                    
#         self.tcn = nn.ModuleList([TemporalBlock(D_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, spec_norm=True),
#                                  *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, spec_norm=True) for i in [1, 2, 4, 8, 16]]])                
#         # self.self_attention = nn.MultiheadAttention(
#         #     embed_dim=hidden_dim,
#         #     num_heads=4,
#         #     dropout=0.05,
#         #     batch_first=True
#         # )

#         self.last = spectral_norm(nn.Conv1d(hidden_dim, 1, kernel_size=1, dilation=1))
#         self.output = nn.Sequential(nn.Linear(n_steps, 1))            

#     def forward(self, x): 
#         x = torch.cat([x], dim=1) 
#         for layer in self.tcn:
#             x = layer(x)
            
#         # x = x.permute(0, 2, 1)
#         # x, _ = self.self_attention(x, x, x)  # Query, Key, Value are all x
#         # x = x.permute(0, 2, 1)
        
#         x = self.last(x)
#         return self.output(x.squeeze()).reshape(-1, 1)

class TCNDiscriminator(nn.Module):
    def __init__(self, D_input_dim, hidden_dim, n_steps, step_size=16):
        super(TCNDiscriminator, self).__init__()                    
        self.step_size = step_size

        self.tcn = nn.ModuleList([TemporalBlock(D_input_dim, hidden_dim, kernel_size=1, stride=1, dilation=1, padding=0, spec_norm=True),
                                 *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, stride=1, dilation=i, padding=i, spec_norm=True) for i in [1, 2, 4, 8]]])                
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,  # Embedding dimension (matches TCN output channel size)
            num_heads=4,   # Number of attention heads
            dropout=0.05,
            batch_first=True
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.last = spectral_norm(nn.Conv1d(hidden_dim, 1, kernel_size=1, dilation=1))
        self.output = nn.Sequential(nn.Linear(n_steps, 1))            
        
        #self.relu = nn.LeakyReLU()  # GELU 활성화 함수로 정의
        self.gelu = nn.GELU()  # GELU 활성화 함수로 정의

        self.dropout = nn.Dropout(0.05)

    def forward(self, x): 
        for layer in self.tcn:
            x = layer(x)
            
        B, C, T = x.shape

        # Reshape z for Self-Attention
        x_attn = x.view(B, C, -1, self.step_size) 
        x_attn = x_attn.permute(0, 3, 2, 1).contiguous() 
        x_attn = x_attn.view(-1, T//self.step_size, C) 

        # Apply Self-Attention
        attn_output, _ = self.self_attention(x_attn, x_attn, x_attn) 

        # Reshape attention output back to original grouped structure
        attn_output = attn_output.view(B, -1,  T//self.step_size, C).permute(0, 3, 1, 2) 

        # Reconstruct full sequence
        attn_output = attn_output.reshape(B, C, T)  # [B, N, T]
        
        # Residual connection
        x = x + self.dropout(attn_output)  # Residual Connection        
        x = self.bn(x)
        #x = self.relu(x)
        x = self.gelu(x)
        
        x = self.last(x)
        return self.output(x.squeeze()).reshape(-1, 1)