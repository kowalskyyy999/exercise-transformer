import torch
import torch.nn as nn
from math import log

class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = PositionalEncoding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids):
        token_embedding = self.token_embedding(input_ids)
        position_embedding = self.position_embedding(input_ids)
        embedding = token_embedding + position_embedding
        embedding = self.layer_norm(embedding)
        return self.dropout(embedding)


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space
        pe = torch.zeros(config.max_len, config.hidden_size).float()
        pe.requires_grad = False
        
        position = torch.arange(0, config.max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, config.hidden_size, 2).float() * -(log(10000.0) / config.hidden_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_dim, n_segments = 2):
        super(SegmentEmbedding, self).__init__(n_segments, embed_dim)