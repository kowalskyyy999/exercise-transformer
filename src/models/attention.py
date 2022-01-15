import torch
import torch.nn as nn
import torch.nn.functional as F 
from math import sqrt

def ScaledDotProductAttn(query, key, value, mask = None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttnHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(AttnHead, self).__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_out = ScaledDotProductAttn(
            self.q(hidden_state), 
            self.k(hidden_state), 
            self.v(hidden_state))
        return attn_out

class MultiHeadAttn(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttn, self).__init__()

        embed_dim = config.hidden_size
        num_heads = config.num_heads
        head_dim = embed_dim // num_heads

        # Build MultiHeadAttention 
        self.heads = nn.ModuleList(
            [AttnHead(embed_dim, head_dim) for _ in range(num_heads)])
        # Build layer linear for output from MLH
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x # output size similar hidden state size

def testing():
    import config as config

    # example sequence length
    seq_length = 100

    # example input
    x = torch.rand(seq_length, config.hidden_size)
    # x size : [100, embed_dim]
    x = x.unsqueeze(0)
    # x size : [1, 100, embed_dim]
    
    # Calling MLH Layer
    MLH = MultiHeadAttn(config)
    
    x_out = MLH(x)

    print(x_out.size(), '\n')
    print(x_out)


if __name__ == '__main__':
    pass



