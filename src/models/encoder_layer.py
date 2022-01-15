import torch
import torch.nn as nn

from attention import MultiHeadAttn
from feed_forward import FeedForwardLayer
from embedding import EmbeddingLayer

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer,self).__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttn(config)
        self.feed_forward = FeedForwardLayer(config)

    def forward(self, x):
        # Apply layer norm and the copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attentio with a skip connection
        x = x + self.attention(hidden_state)
        # Apply Feed Forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.embedding = EmbeddingLayer(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)
                                    for _ in range(config.num_encoders_layers)])
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x
