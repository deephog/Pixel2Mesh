import math

import torch
import torch.nn as nn

from utils.tensor import dot


class GConv(nn.Module):
    """Simple GCN layer

    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        #self.adj_mat = nn.Parameter(adj_mat, requires_grad=False)
        self.weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        # Following https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch/blob/a0ae88c4a42eef6f8f253417b97df978db842708/model/gcn_layers.py#L45
        # This seems to be different from the original implementation of P2M
        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.loop_weight.data)

    def forward(self, inputs, adj_mat):

        support = torch.matmul(inputs, self.weight)
        support_loop = torch.matmul(inputs, self.loop_weight)
        output = dot(torch.squeeze(adj_mat), support, False) + support_loop
        if self.bias is not None:
            ret = output + self.bias
        else:
            ret = output
        return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GConvGated(nn.Module):
    """Simple GCN layer

    Similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GConvGated, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        #self.adj_mat = nn.Parameter(adj_mat, requires_grad=False)
        self.weight_ir = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        self.weight_hr = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))

        self.weight_iz = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        self.weight_hz = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))

        self.weight_hn = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        self.weight_in = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))

        self.loop_weight = nn.Parameter(torch.zeros((in_features, out_features), dtype=torch.float))
        # Following https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch/blob/a0ae88c4a42eef6f8f253417b97df978db842708/model/gcn_layers.py#L45
        # This seems to be different from the original implementation of P2M

        self.bias_ir = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        self.bias_hr = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))

        self.bias_iz = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        self.bias_hz = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))

        self.bias_hn = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        self.bias_in = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))

        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ir.data)
        nn.init.xavier_uniform_(self.weight_hr.data)
        nn.init.xavier_uniform_(self.weight_iz.data)
        nn.init.xavier_uniform_(self.weight_hz.data)
        nn.init.xavier_uniform_(self.weight_in.data)
        nn.init.xavier_uniform_(self.weight_hn.data)

        nn.init.xavier_uniform_(self.loop_weight.data)

    def forward(self, inputs, hidden, adj_mat):

        r = torch.sigmoid(torch.matmul(inputs, self.weight_ir) + self.bias_ir + torch.matmul(hidden, self.weight_hr) + self.bias_hr)
        z = torch.sigmoid(torch.matmul(inputs, self.weight_iz) + self.bias_iz + torch.matmul(hidden, self.weight_hz) + self.bias_hz)
        n = torch.relu(torch.matmul(inputs, self.weight_in) + self.bias_in + r * (torch.matmul(hidden, self.weight_hn) + self.bias_hn))
        next_hidden = (1 - z) * n + z * hidden
        support_loop = torch.matmul(inputs, self.loop_weight)
        output = dot(torch.squeeze(adj_mat), next_hidden, False) + support_loop
        if self.bias is not None:
            ret = output + self.bias
        else:
            ret = output
        return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'