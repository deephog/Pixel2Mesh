import os
import random
from logging import Logger
import time
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import mesh_load
from functions.base import CheckpointRunner
from models.p2m import P2MHuman, P2MHumanGRU
from multi_data import get_loader
# from utils.mesh import Ellipsoid
from torch_geometric import utils
import open3d as o3d
#from utils.vis.renderer import MeshRenderer


class Predictor(CheckpointRunner):

    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        self.gpu_inference = self.options.num_gpus > 0
        if self.gpu_inference == 0:
            raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                      "might be fixed in the future.")
            # self.logger.warning("Render part would be disabled since you are using CPU. "
            #                     "Neural renderer requires GPU to run. Please use other softwares "
            #                     "or packages to view .obj file generated.")

        if self.options.model.name == "pixel2mesh":
            # create ellipsoid
            #self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
            # create model
            cameras = mesh_load.camera_setting_loader(
                os.path.join(self.options.dataset.data_dir, 'default_camera_setting.xml'))

            self.model = P2MHuman(self.options.model, cameras=cameras)
            if self.gpu_inference:
                self.model.cuda()
                # create renderer
                # self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
                #                              self.options.dataset.mesh_pos)
        else:
            raise NotImplementedError("Currently the predictor only supports pixel2mesh")

    def models_dict(self):
        return {'model': self.model}

    def predict_step(self, input_batch, index, last_verts=None, last_matrix=None, last_tri=None):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            image, this_mesh, next_mesh, names, sample_views, mask, depth = input_batch
            # print(names[0])
            # Grab data from the batch
            images = torch.squeeze(image, dim=0).cuda()
            depth = torch.squeeze(depth, dim=0).cuda()
            if last_verts == None:
                verts = torch.permute(torch.squeeze(this_mesh['norm_vert']), (1, 0)).cuda()
                edges = torch.squeeze(this_mesh['edge']).cuda()
                matrix = utils.to_dense_adj(edges)
                tri = torch.squeeze(this_mesh['face']).numpy()
            else:
                verts, matrix, tri = last_verts, last_matrix, last_tri

            start_time = time.time()
            out = self.model(depth, verts, matrix, sample_views, recover=[None, 3000.])
            print(time.time()-start_time)
            self.render(input_batch, out, tri, index)
            return out, matrix, tri


    def predict_seq(self, input_batch):
        self.model.eval()

        # Run inference
        with torch.no_grad():
            # image, this_mesh, next_mesh, names, sample_views, mask, depth = input_batch
            # print(names[0])
            # Grab data from the batch

            seq = input_batch

            num_steps = len(seq)

            # print(names[0])
            # Grab data from the batch
            # images = torch.squeeze(image, dim=0).cuda()
            images = []
            sample_views = []
            for s in seq:
                images.append(torch.squeeze(s['depth'], dim=0).cuda())
                sample_views.append(s['sample_views'])
            verts = torch.squeeze(seq[0]['this_mesh']['norm_vert']).cuda()
            verts = torch.transpose(verts, 0, 1)
            edges = torch.squeeze(seq[0]['this_mesh']['edge']).cuda()
            tri = torch.squeeze(seq[0]['this_mesh']['face']).numpy()
            # adj = edgelist2adjacency(edges.numpy())

            matrix = utils.to_dense_adj(edges)

            start_time = time.time()
            out = self.model(images, verts, matrix, sample_views, recover=[None, 3000.])
            print(time.time()-start_time)
            for index in range(num_steps):
                self.render(input_batch, out[index], tri, index)
            return out, matrix, tri


    def predict_series(self, list_batch):
        last_verts = None
        last_matrix = None
        last_tri = None
        index = 0
        for b in list_batch:
            last_verts, last_matrix, last_tri = self.predict_step(b, index, last_verts, last_matrix, last_tri)
            last_verts = torch.transpose(last_verts["pred_coord"][-1], 0, 1)/3000.
            index +=1


    def predict(self):
        self.logger.info("Running predictions...")

        # predict_data_loader = DataLoader(self.dataset,
        #                                  batch_size=self.options.test.batch_size,
        #                                  pin_memory=self.options.pin_memory,
        #                                  collate_fn=self.dataset_collate_fn)

        predict_data_loader = get_loader(data_root=self.options.dataset.data_dir,
                                            mesh_root='/home/hypevr/nas/2021/09/0927/deform_data_set_02/LowRes/mesh_with_normal/',
                                            lap_root='/home/hypevr/nas/2021/09/0927/deform_data_set_02/LowRes/laplacian/',
                                            batchsize=1,
                                            trainsize=(256, 448),
                                            origsize=None,
                                            n_views=8,
                                            shuffle=False,
                                            num_workers=20,
                                            pin_memory=True,
                                            fake_back_rate=0,
                                            back_dir=None,
                                            back_img=None,
                                            pure_back_rate=0,
                                            with_plate=False,
                                            examine_mode=False,
                                            trimap_dir=None,
                                            of_list=None,
                                            with_gray=0,
                                            mask_ext='.jpg')

        n = 0
        nmax = 8
        n_list = []
        for step, batch in enumerate(predict_data_loader):
            # self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))

            # if self.gpu_inference:
            #     # Send input to GPU
            #     batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            #n_list.append(batch)
            #n += 1
            self.predict_seq(batch)
            # if n == 8:
            #     #self.predict_step(batch)
            #     self.predict_series(n_list)
            #     n = 0
            #     n_list = []
            input('wait')

    def render(self, inputs, outputs, pred_tri, index):
        #image, this_mesh, next_mesh, names, sample_views, mask, depth = inputs
        next_predpc = outputs["pred_coord"][-1].cpu().numpy()
        #pred_tri = torch.squeeze(this_mesh['face']).numpy()
        pred_tri = pred_tri##.cpu().numpy()

        this_vert = torch.squeeze(inputs[0]['this_mesh']['vert']).numpy()
        next_vert = torch.squeeze(inputs[index]['next_mesh']['vert']).numpy()
        next_tri = torch.squeeze(inputs[index]['next_mesh']['face']).numpy()

        next_pred = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(next_predpc.astype(np.float64)), triangles=o3d.utility.Vector3iVector(pred_tri.astype(np.int32)))
        this_geo = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(this_vert.astype(np.float64)), triangles=o3d.utility.Vector3iVector(pred_tri.astype(np.int32)))
        next_geo = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(next_vert.astype(np.float64)), triangles=o3d.utility.Vector3iVector(next_tri.astype(np.int32)))

        #print(names[0])

        o3d.io.write_triangle_mesh(os.path.join('output', str(index) + '_pred_mesh.ply'), next_pred) #names[0][0][:-4]
        o3d.io.write_triangle_mesh(os.path.join('output', str(index) + '_this_mesh.ply'), this_geo)
        o3d.io.write_triangle_mesh(os.path.join('output', str(index) + '_next_mesh.ply'), next_geo)

        # this_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(next_predpc.astype(np.float64)),
        #                                       triangles=o3d.utility.Vector3iVector(pred_tri.astype(np.int32)))


        #next_gt = o3d.geometry.TriangleMesh(vertices=next_predpc, triangles=next_tri)

    def render_back(self, inputs, outputs, pred_tri, index):
        image, this_mesh, next_mesh, names, sample_views, mask, depth = inputs
        next_predpc = outputs["pred_coord"][-1].cpu().numpy()
        #pred_tri = torch.squeeze(this_mesh['face']).numpy()
        pred_tri = pred_tri##.cpu().numpy()

        this_vert = torch.squeeze(this_mesh['vert']).numpy()

        next_vert = torch.squeeze(next_mesh['vert']).numpy()
        next_tri = torch.squeeze(next_mesh['face']).numpy()

        next_pred = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(next_predpc.astype(np.float64)), triangles=o3d.utility.Vector3iVector(pred_tri.astype(np.int32)))
        this_geo = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(this_vert.astype(np.float64)), triangles=o3d.utility.Vector3iVector(pred_tri.astype(np.int32)))
        next_geo = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(next_vert.astype(np.float64)), triangles=o3d.utility.Vector3iVector(next_tri.astype(np.int32)))

        #print(names[0])

        o3d.io.write_triangle_mesh(os.path.join('output', str(index) + '_pred_mesh.ply'), next_pred) #names[0][0][:-4]
        o3d.io.write_triangle_mesh(os.path.join('output', str(index) + '_this_mesh.ply'), this_geo)
        o3d.io.write_triangle_mesh(os.path.join('output', str(index) + '_next_mesh.ply'), next_geo)

        # this_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(next_predpc.astype(np.float64)),
        #                                       triangles=o3d.utility.Vector3iVector(pred_tri.astype(np.int32)))


        #next_gt = o3d.geometry.TriangleMesh(vertices=next_predpc, triangles=next_tri)


    def save_inference_results(self, inputs, outputs):
        if self.options.model.name == "pixel2mesh":
            batch_size = inputs["images"].size(0)
            for i in range(batch_size):
                basename, ext = os.path.splitext(inputs["filepath"][i])
                mesh_center = np.mean(outputs["pred_coord_before_deform"][0][i].cpu().numpy(), 0)
                verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
                for k, vert in enumerate(verts):
                    meshname = basename + ".%d.obj" % (k + 1)
                    vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                    mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[k]))
                    np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")

                if self.gpu_inference:
                    # generate gif here

                    color_repo = ['light_blue', 'purple', 'orange', 'light_yellow']

                    rot_degree = 10
                    rot_radius = rot_degree / 180 * np.pi
                    rot_matrix = np.array([
                        [np.cos(rot_radius), 0, -np.sin(rot_radius)],
                        [0., 1., 0.],
                        [np.sin(rot_radius), 0, np.cos(rot_radius)]
                    ])
                    writer = imageio.get_writer(basename + ".gif", mode='I')
                    color = random.choice(color_repo)
                    for _ in tqdm(range(360 // rot_degree), desc="Rendering sample %d" % i):
                        image = inputs["images_orig"][i].cpu().numpy()
                        ret = image
                        for k, vert in enumerate(verts):
                            vert = rot_matrix.dot((vert - mesh_center).T).T + mesh_center
                            rend_result = self.renderer.visualize_reconstruction(None,
                                                                                 vert + \
                                                                                 np.array(
                                                                                     self.options.dataset.mesh_pos),
                                                                                 self.ellipsoid.faces[k],
                                                                                 image,
                                                                                 mesh_only=True,
                                                                                 color=color)
                            ret = np.concatenate((ret, rend_result), axis=2)
                            verts[k] = vert
                        ret = np.transpose(ret, (1, 2, 0))
                        writer.append_data((255 * ret).astype(np.uint8))
                    writer.close()
