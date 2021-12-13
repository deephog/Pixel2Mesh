import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Threshold

class SampleHypothesis(nn.Module):
    def __init__(self, options):
        super(SampleHypothesis, self).__init__()
        self.sample_delta = torch.from_numpy(options.sample_coord).type(torch.float32).cuda()
        # mult = torch.tensor([1], dtype=torch.float32).cuda()
        # self.magnify = Variable(mult)

    def forward(self, mesh_coords, mag):
        """
        Local Grid Sample for fast matching init mesh
        :param mesh_coords:
        [N,S,3] ->[NS,3] for projection
        :return: sample_points_per_vertices: [NS, 3]
        """

        center_points = torch.unsqueeze(mesh_coords, dim=1)
        center_points = torch.tile(center_points, [1, 43, 1])

        delta = torch.unsqueeze(self.sample_delta, dim=0)
        delta = torch.multiply(delta, mag)

        sample_points_per_vertices = torch.add(center_points, delta)

        outputs = torch.reshape(sample_points_per_vertices, [-1, 3])
        return outputs


class DeformationReasoning(nn.Module):
    def __init__(self, input_dim, output_dim, options, gcn_block=-1, feat_dim=None):
        super(DeformationReasoning, self).__init__()
        self.delta_coord = torch.from_numpy(options.sample_coord).type(torch.float32).cuda()
        self.s = 43
        self.f = input_dim
        self.hidden_dim = 192

        self.local_conv1 = LocalGConv(input_dim=input_dim, output_dim=self.hidden_dim, options=options)
        self.local_conv2 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, options=options)
        self.local_conv3 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, options=options)
        self.local_conv4 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, options=options)
        self.local_conv5 = LocalGConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim, options=options)
        self.local_conv6 = LocalGConv(input_dim=self.hidden_dim, output_dim=1, options=options)

    def forward(self, inputs):
        proj_feat, prev_coord = inputs[0], inputs[1]

        x = proj_feat  # NS, F
        x = torch.reshape(x, [-1, self.s, self.f])  # N,S,F
        x1 = self.local_conv1(x)

        x2 = self.local_conv2(x1)
        x3 = torch.add(self.local_conv3(x2), x1)
        x4 = self.local_conv4(x3)
        x5 = torch.add(self.local_conv5(x4), x3)
        x6 = self.local_conv6(x5)  # N, S, 1
        score = torch.nn.functional.softmax(x6, dim=1)  # N, S, 1


        #tf.summary.histogram('score', score)
        delta_coord = score * torch.unsqueeze(self.delta_coord, dim=0)

        next_coord = torch.sum(delta_coord, dim=1)
        next_coord += prev_coord
        return next_coord



class LocalGConv(nn.Module):
    def __init__(self, input_dim, output_dim, options, dropout=False, act=torch.nn.LeakyReLU(), bias=True):
        super(LocalGConv, self).__init__()

        if dropout:
            self.dropout = options.dropout
        else:
            self.dropout = 0.

        self.act = act
        self.support = []
        for i in options.sample_adj:
            self.support.append(torch.from_numpy(i).type(torch.float32).cuda())

        self.bias = bias
        self.local_graph_vert = 43
        self.drop = torch.nn.Dropout(1 - self.dropout)
        self.output_dim = output_dim
        self.vars = {}

        for i in range(len(self.support)):
            self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
        if self.bias:
            self.vars['bias'] = zeros([output_dim], name='bias')

        # if self.logging:
        #     self._log_vars()

    def forward(self, inputs):
        x = inputs  # N, S, VF
        # dropout
        #x = self.drop(x)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            pre_sup = torch.einsum('ijk,kl->ijl', x, self.vars['weights_' + str(i)])
            support = torch.einsum('ij,kjl->kil', self.support[i], pre_sup)
            supports.append(support)
        output = sum(supports)
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)



def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(-init_range, init_range, shape)
    initial = torch.from_numpy(initial).type(torch.float32)
    initial = torch.nn.Parameter(initial).cuda()
    #initial = Variable(initial, name=name)
    return initial


def zeros(shape, name=None):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32)
    return torch.nn.Parameter(initial).cuda()