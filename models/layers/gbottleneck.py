import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm

from models.layers.gconv import GConv


class GResBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, activation=None):
        super(GResBlock, self).__init__()

        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim)
        self.conv2 = GConv(in_features=hidden_dim, out_features=in_dim)
        self.activation =nn.LeakyReLU() if activation else None

    def forward(self, dict):
        inputs = dict['x']
        adj_mat = dict['adj']
        x = self.conv1(inputs, adj_mat)

        if self.activation:
            x = self.activation(x)
        x = self.conv2(x, adj_mat)

        if self.activation:
            x = self.activation(x)

        out = (inputs + x) * 0.5
        return {'x':out, 'adj':adj_mat}


class GBottleneck(nn.Module):

    def __init__(self, block_num, in_dim, hidden_dim, out_dim, activation=None):
        super(GBottleneck, self).__init__()

        resblock_layers = [GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, activation=activation)
                           for _ in range(block_num)]
        self.blocks = nn.Sequential(*resblock_layers)
        self.conv1 = GConv(in_features=in_dim, out_features=hidden_dim)

        self.conv2 = GConv(in_features=hidden_dim, out_features=out_dim)

        self.activation = nn.LeakyReLU() if activation else None

    def forward(self, inputs, adj_mat):
        x = self.conv1(inputs, adj_mat)
        if self.activation:
            x = self.activation(x)
        dict = {'x': x, 'adj': adj_mat}
        x_hidden = self.blocks(dict)
        x_out = self.conv2(x_hidden['x'], adj_mat)
        # print(x_out.shape)

        return x_out, x_hidden
