import trimesh
import torch
#from torch_geometric import utils

import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import time
from sknetwork.utils import edgelist2adjacency
#from pytorch3d.structures import Meshes
#from pytorch3d.loss import chamfer_distance, mesh_normal_consistency, mesh_edge_loss, mesh_laplacian_smoothing

# def load_mesh(file, sample_size=None):
#     f = trimesh.load(file)
#     if sample_size:
#         vertices, faces = trimesh.sample.sample_surface(f, sample_size)
#     else:
#         vertices = f.metadata['ply_raw']['vertex']['data']
#         faces = f.metadata['ply_raw']['face']['data']['vertex_indices']
#
#     edges = trimesh.geometry.faces_to_edges(faces).astype(np.int64)
#     #edge_tensor = torch.from_numpy(edges).type(torch.int64)
#     #matrix = utils.to_dense_adj(edge_tensor)
#     vert_array = np.concatenate([vertices['x'], vertices['y'], vertices['z']], axis=1)
#     #vert_tensor = torch.from_numpy(vert_array).type(torch.FloatTensor)
#     del faces, vertices, f
#     return {'vert': vert_array, 'edge': edges}


def load_mesh(file, sample_size=None):
    mesh = o3d.io.read_triangle_mesh(file)
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int64)
    #vol = mesh.get_volume()
    #mesh.compute_vertex_normals()
    norms = np.asarray(mesh.vertex_normals).astype(np.float32)
    edges = trimesh.geometry.faces_to_edges(faces).astype(np.int64)

    # diag0 = verts.max(axis=0)
    # diag1 = verts.min(axis=0)

    #diag_len = np.array([3000.])#np.sqrt(np.sum(np.square(diag1 - diag0)))

    #cent = np.mean(verts, axis=0, keepdims=True)
    norm_verts = verts / 3000.

    return {'vert': verts, 'edge': edges, 'norm': norms, 'face': faces, 'norm_vert': norm_verts} #, 'volume': vol

def load_lap(file, mesh):
    lap = np.load(file)
    mesh['lap'] = lap
    return mesh


def camera_setting_loader(xml_dir):
    content = ET.parse(xml_dir)
    list = content.find('camera_settings')  # .attrib('id')
    camera_list = list.findall('camera')
    camera_id = []
    intrinsics_rgb = []
    intrinsics_ir = []
    extrinsics_rgb = []
    extrinsics_ir = []
    distortion_rgb = []
    distortion_ir = []
    for c in camera_list:
        camera_id.append(c.attrib['id'])
        dual = c.findall('calibration')
        intrinsics_rgb.append(dual[0].find('intrinsics').attrib)
        intrinsics_ir.append(dual[1].find('intrinsics').attrib)
        extrinsics_rgb.append(dual[0].find('extrinsics').attrib)
        extrinsics_ir.append(dual[1].find('extrinsics').attrib)
        distortion_rgb.append(dual[0].find('distortion').attrib)
        distortion_ir.append(dual[1].find('distortion').attrib)

    cameras = {'camera_id': camera_id, 'int_rgb': intrinsics_rgb, 'int_ir': intrinsics_ir, 'ext_rgb': extrinsics_rgb,
               'ext_ir': extrinsics_ir, 'dstn_rgb': distortion_rgb, 'dstn_ir': distortion_ir}
    return cameras


def main():
    file = 'pred_mesh_fixed.ply'
    file2 = '/home/hypevr/Desktop/projects/Pixel2Mesh_bak/output/issue/3_pred_mesh.ply'

    lap_f = '/home/hypevr/nas/2021/09/0927/deform_data_set_02/LowRes/laplacian/laplacian000222.numpy'
    # mesh = load_mesh(file)
    # mesh2 = load_mesh(file2)

    # mesh = load_lap(lap_f, mesh)
    mesh = trimesh.load_mesh(file)
    mesh2 = trimesh.load_mesh(file2)
    trimesh.repair.fill_holes(mesh)
    mesh = mesh.as_open3d
    vol = mesh.get_volume()

    print(vol)

    # print(mesh['volume'])  94353642.99232812
    #
    # edges = mesh['edge']
    # edges = torch.from_numpy(edges).type(torch.int64)
    #
    # lap = utils.get_laplacian(torch.transpose(edges, 0, 1))
    #
    #
    # print(mesh2['norm'])
    # pymesh = Meshes([torch.from_numpy(mesh['vert']), torch.from_numpy(mesh2['vert'])], [torch.from_numpy(mesh['face']), torch.from_numpy(mesh2['face'])])
    # #pymesh.#print(pymesh.)
    # pymesh.cuda()
    # chamf = chamfer_distance(x=torch.unsqueeze(torch.from_numpy(mesh['vert']).type(torch.float32), 0), y=torch.unsqueeze(torch.from_numpy(mesh2['vert']).type(torch.float32), 0),
    #                          x_normals=torch.unsqueeze(torch.from_numpy(mesh['norm']).type(torch.float32), 0), y_normals=torch.unsqueeze(torch.from_numpy(mesh2['norm']).type(torch.float32), 0)
    #                          )
    # lap = mesh_laplacian_smoothing(pymesh)
    # edg = mesh_edge_loss(pymesh)
    # nom = mesh_normal_consistency(pymesh)
    # print(chamf)
    #
    # #vertices, edge_tensor, matrix = load_mesh(file)
    # #vert_array = np.concatenate([vertices['x'], vertices['y'], vertices['z']], axis=1)
    # #print(vert_array)

if __name__ == "__main__":
    main()
