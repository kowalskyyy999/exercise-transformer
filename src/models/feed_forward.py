import torch
import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.ff_size)
        self.linear2 = nn.Linear(config.ff_size, config.hidden_size)
        self.activ = activation_func(config.activation_func)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.activ(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return x

def activation_func(activ):
    if activ == "relu":
        return nn.ReLU()
    elif activ == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError

def testing():
    import config as config
    ff_layer = FeedForwardLayer(config)
    print(ff_layer)

if __name__ == '__main__':
    pass