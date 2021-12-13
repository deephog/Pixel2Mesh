# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
#import tensorflow as tf
import numpy as np
import torch

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = 2*scale * torch.rand(shape) - scale
    return torch.autograd.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    scale = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = 2*scale * torch.rand(shape) - scale
    return torch.autograd.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32)
    return torch.autograd.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = torch.ones(shape, dtype=torch.float32)
    return torch.autograd.Variable(initial, name=name)
