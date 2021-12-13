# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import numpy as np
import torch
import cv2

def construct_feed_dict(pkl, placeholders):
    coord = pkl['coord']
    pool_idx = pkl['pool_idx']
    faces = pkl['faces']
    lape_idx = pkl['lape_idx']

    edges = []
    for i in range(1, 4):
        adj = pkl['stage{}'.format(i)][1]
        edges.append(adj[0])

    feed_dict = dict()
    feed_dict.update({placeholders['features']: coord})
    feed_dict.update({placeholders['edges'][i]: edges[i] for i in list(range(len(edges)))})
    feed_dict.update({placeholders['faces'][i]: faces[i] for i in list(range(len(faces)))})
    feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in list(range(len(pool_idx)))})
    feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in list(range(len(lape_idx)))})
    feed_dict.update({placeholders['support1'][i]: pkl['stage1'][i] for i in list(range(len(pkl['stage1'])))})
    feed_dict.update({placeholders['support2'][i]: pkl['stage2'][i] for i in list(range(len(pkl['stage2'])))})
    feed_dict.update({placeholders['support3'][i]: pkl['stage3'][i] for i in list(range(len(pkl['stage3'])))})

    for k in range(3):
        feed_dict.update({placeholders['faces_triangle'][k]: pkl['faces_triangle'][k]})

    feed_dict.update({placeholders['sample_coord']: pkl['sample_coord']})

    feed_dict.update({placeholders['sample_adj'][i]: pkl['sample_cheb_dense'][i] for i in range(len(pkl['sample_cheb_dense']))})

    return feed_dict


def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = torch.mean(x, axis, True)
    devs_squared = torch.square(x - m)
    return torch.mean(devs_squared, axis, keepdims)


def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return torch.sqrt(reduce_var(x, axis=axis, keepdims=keepdims) + 1e-6)

# -------------------------------------------------------------------
# cameras
# -------------------------------------------------------------------


def normal(v):
    norm = torch.norm(v)
    if norm == 0:
        return v
    return torch.divide(v, norm)


def cameraMat(param):
    theta = param[0] * np.pi / 180.0
    camy = param[3] * torch.sin(param[1] * np.pi / 180.0)
    lens = param[3] * torch.cos(param[1] * np.pi / 180.0)
    camx = lens * torch.cos(theta)
    camz = lens * torch.sin(theta)
    Z = torch.stack([camx, camy, camz])

    x = camy * torch.cos(theta + np.pi)
    z = camy * torch.sin(theta + np.pi)
    Y = torch.stack([x, lens, z])
    X = torch.cross(Y, Z)

    cm_mat = torch.stack([normal(X), normal(Y), normal(Z)])
    return cm_mat, Z


def camera_trans(camera_metadata, xyz):
    # c, o = cameraMat(camera_metadata)

    c = np.expand_dims(
        np.array([float(camera_metadata['rx']), float(camera_metadata['ry']), float(camera_metadata['rz'])]), axis=0)

    o = torch.tensor([float(camera_metadata['tx']), float(camera_metadata['ty']), float(camera_metadata['tz'])]).type(torch.float32).cuda()
    o = torch.unsqueeze(o, 0)

    trans_mat = np.zeros((3, 3))
    cv2.Rodrigues(c, trans_mat)
    trans_mat = torch.from_numpy(trans_mat).cuda().type(torch.float32)

    #pt_trans = points
    pt_trans = torch.matmul(torch.transpose(xyz, 0, 1), torch.transpose(trans_mat, 0, 1)) + o
    return pt_trans


def camera_trans_inv(camera_metadata, xyz):
    #c, o = cameraMat(camera_metadata)
    c = np.expand_dims(np.array([float(camera_metadata['rx']), float(camera_metadata['ry']), float(camera_metadata['rz'])]), axis=0)
    o = torch.unsqueeze(torch.tensor([float(camera_metadata['tx']), float(camera_metadata['ty']), float(camera_metadata['tz'])]), 0).cuda().type(torch.float32)
    trans_mat = np.zeros((3, 3))
    cv2.Rodrigues(c, trans_mat)
    trans_mat = torch.from_numpy(trans_mat).cuda().type(torch.float32)
    inv_xyz = (torch.matmul(torch.transpose(xyz, 0, 1), torch.linalg.inv(torch.transpose(trans_mat, 0, 1)))) + o#

    return torch.transpose(inv_xyz, 0, 1)


def load_demo_image(demo_image_list):
    imgs = np.zeros((3, 224, 224, 3))
    for idx, demo_img_path in enumerate(demo_image_list):
        img = cv2.imread(demo_img_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            img[np.where(img[:, :, 3] == 0)] = 255
        img = cv2.resize(img, (224, 224))
        img_inp = img.astype('float32') / 255.0
        imgs[idx] = img_inp[:, :, :3]
    return imgs


def th_gather_nd(x, coords):
    x = x.contiguous()
    str = torch.FloatTensor(x.stride()).cuda()
    inds = coords.mv(str).type(torch.long)
    x_gather = torch.index_select(th_flatten(x), 0, inds)
    return x_gather


def th_flatten(x):
    """Flatten tensor"""
    return x.contiguous().view(-1)


def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices

    returns: tensor shaped [m_1, m_2, m_3, m_4]

    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices

    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1)  # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)

