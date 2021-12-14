import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.chamfer_wrapper import ChamferDist
#from pytorch3d.structures import Meshes
#from pytorch3d.loss import chamfer_distance, mesh_normal_consistency, mesh_edge_loss, mesh_laplacian_smoothing
#from pytorch3d.ops import estimate_pointcloud_normals


class P2MLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()
        # self.laplace_idx = nn.ParameterList([
        #     nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx])
        # self.edges = nn.ParameterList([
        #     nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])

    def edge_regularization(self, pred, edges):
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        # if len(pred.shape) == 2:
        #     pred = torch.unsqueeze(pred, dim=0)
        edges = torch.squeeze(edges)
        #print(pred.shape, edges.shape)
        return self.l2_loss(pred[edges[:, 0], :], pred[edges[:, 1], :]) * pred.size(-1)

    def edge_loss(self, pred_pc, pred_edges, next_pc, next_edges):
        reg1 = self.edge_regularization(pred_pc, pred_edges)
        reg2 = self.edge_regularization(next_pc, next_edges)
        loss = self.l2_loss(reg1, reg2)
        return loss

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """
        lap_idx = torch.squeeze(lap_idx)
        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        #inputs = torch.transpose(inputs, 0, 1)
        #print(inputs.shape, all_valid_indices.shape)
        vertices = inputs[all_valid_indices.long(), :]
        vertices[invalid_mask.long(), :] = 0

        neighbor_sum = torch.sum(vertices, 1).cuda()
        neighbor_count = lap_idx[:, -1].float().cuda()

        neighbor_count[neighbor_count == 0] = 1
        #print(torch.unique(neighbor_count))

        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(self, input1, input2, lap_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        input1 = torch.transpose(input1, 0, 1)

        lap1 = self.laplace_coord(input1, lap_idx)

        lap2 = self.laplace_coord(input2, lap_idx)


        laplace_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)


        #move_loss = self.l2_loss(input1, input2) * input1.size(-1)
        return laplace_loss#, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):
        adj_list = torch.squeeze(adj_list)
        edges = pred_points[adj_list[:, 0], :] - pred_points[adj_list[:, 1], :].cuda()

        F.normalize(edges, dim=1)
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])

        normals = F.normalize(torch.squeeze(nearest_normals)[adj_list[:, 0], :], dim=1).cuda()
        cosine = torch.abs(torch.sum(edges * normals, 1))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def forward(self, outputs, this_mesh, next_mesh, gt_images):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """

        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss = 0., 0., 0., 0., 0.

        #gt_coord, gt_normal, gt_images = targets["points"], targets["normals"], targets["images"]
        gt_coord, gt_normal, gt_faces, gt_edges = next_mesh["vert"], next_mesh["norm"], next_mesh["face"], next_mesh['edge']
        this_edges, this_faces, this_norms, this_lap = this_mesh['edge'], this_mesh['face'], this_mesh['norm'], this_mesh['lap']
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]
        image_loss = 0.
        if outputs["reconst"] is not None and self.options.weights.reconst != 0:
            image_loss = self.image_loss(gt_images, outputs["reconst"])


        #mesh_pred = Meshes(torch.unsqueeze(pred_coord, 0), this_faces.cuda())
        #mesh_gt = Meshes(gt_coord, gt_faces)
        #pred_norm = estimate_pointcloud_normals(torch.unsqueeze(pred_coord, 0))

        #chamfer_loss, normal_loss = chamfer_distance(x=torch.unsqueeze(pred_coord, 0), y=gt_coord.cuda(), batch_reduction='mean', point_reduction='mean')
        #normal_loss = 0
        # dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, pred_coord)
        # chamfer_loss = (torch.mean(dist1) + torch.mean(dist2))
        # normal_loss = self.normal_loss(gt_normal, idx2, pred_coord, this_edges)
        # edge_loss = mesh_edge_loss(mesh_pred)
        # lap = mesh_laplacian_smoothing(mesh_pred)


        for i in range(2):
            dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, torch.unsqueeze(pred_coord[i], 0))
            chamfer_loss += self.options.weights.chamfer[i] * (torch.mean(dist1) +
                                                               self.options.weights.chamfer_opposite * torch.mean(dist2))

            normal_loss += self.normal_loss(gt_normal, idx2, pred_coord[i], this_edges)
            # print(pred_coord[i].shape, this_edges.shape)
            # print(gt_coord.shape, gt_edges.shape)
            edge_loss += self.edge_loss(pred_coord[i], this_edges, torch.squeeze(gt_coord).cuda(), gt_edges.cuda())#self.edge_regularization(pred_coord[i], this_edges)


        #print(pred_coord[1])
        # lap = self.laplace_regularization(pred_coord_before_deform, pred_coord[1], this_lap)
        # lap_loss += lap
            #move_loss += move

        #print(move_loss)

        loss = chamfer_loss + image_loss * self.options.weights.reconst + \
               self.options.weights.laplace * lap_loss + \
               self.options.weights.move * move_loss + \
               self.options.weights.edge * edge_loss + \
               self.options.weights.normal * normal_loss #

        loss = loss * self.options.weights.constant


        # print('cham', chamfer_loss)
        # print('edge', edge_loss)
        # print('lap', lap_loss)
        # print('norm', normal_loss)
        # print('total', loss)
        #
        # print('weighted_lap', self.options.weights.laplace * lap_loss,
        #       'weighted_edge', self.options.weights.edge * edge_loss,
        #       'weighted_norm', self.options.weights.normal * normal_loss)

        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            "loss_move": move_loss,
            "loss_normal": normal_loss,
        }



class LaplacianLoss(object):
    """
    Encourages minimal mean curvature shapes.
    """

    def __init__(self, faces, vert, toref=True):
        # Input:
        #  faces: B x F x 3
        self.toref = toref
        from laplacian import Laplacian
        # V x V
        self.laplacian = Laplacian(faces)
        self.Lx = None
        tmp = self.laplacian(vert)
        self.curve_gt = torch.norm(tmp.view(-1, tmp.size(2)), p=2, dim=1).float()
        if not self.toref:
            self.curve_gt = self.curve_gt * 0

    def __call__(self, verts):
        self.Lx = self.laplacian(verts)
        # Reshape to BV x 3
        Lx = self.Lx.view(-1, self.Lx.size(2))
        loss = (torch.norm(Lx, p=2, dim=1).float() - self.curve_gt).mean()
        return loss
