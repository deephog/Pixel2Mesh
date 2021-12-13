import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Threshold
from models.layers.modules.tools import *
#from torchsample.utils import th_gather_nd




class GProjection(nn.Module):
    """
    Graph Projection layer, which pool 2D features to mesh

    The layer projects a vertex of the mesh to the 2D image and use
    bi-linear interpolation to get the corresponding feature.
    """

    def __init__(self, mesh_pos, camera_f, camera_c, bound=0, tensorflow_compatible=False):
        super(GProjection, self).__init__()
        self.mesh_pos, self.camera_f, self.camera_c = mesh_pos, camera_f, camera_c
        self.threshold = None
        self.bound = 0
        self.tensorflow_compatible = tensorflow_compatible
        if self.bound != 0:
            self.threshold = Threshold(bound, bound)

    def bound_val(self, x):
        """
        given x, return min(threshold, x), in case threshold is not None
        """
        if self.bound < 0:
            return -self.threshold(-x)
        elif self.bound > 0:
            return self.threshold(x)
        return x

    @staticmethod
    def image_feature_shape(img):
        return np.array([img.size(-1), img.size(-2)])

    def project_tensorflow(self, x, y, img_size, img_feat):
        x = torch.clamp(x, min=0, max=img_size[1] - 1)
        y = torch.clamp(y, min=0, max=img_size[0] - 1)

        # it's tedious and contains bugs...
        # when x1 = x2, the area is 0, therefore it won't be processed
        # keep it here to align with tensorflow version
        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        weights = torch.mul(x2.float() - x, y2.float() - y)
        Q11 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2.float() - x, y - y1.float())
        Q12 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q12, 0, 1))

        weights = torch.mul(x - x1.float(), y2.float() - y)
        Q21 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1.float(), y - y1.float())
        Q22 = torch.mul(weights.unsqueeze(-1), torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22
        return output

    def forward(self, resolution, img_features, inputs):
        half_resolution = (resolution - 1) / 2
        camera_c_offset = np.array(self.camera_c) - half_resolution
        # map to [-1, 1]
        # not sure why they render to negative x
        positions = inputs + torch.tensor(self.mesh_pos, device=inputs.device, dtype=torch.float)
        w = -self.camera_f[0] * (positions[:, :, 0] / self.bound_val(positions[:, :, 2])) + camera_c_offset[0]
        h = self.camera_f[1] * (positions[:, :, 1] / self.bound_val(positions[:, :, 2])) + camera_c_offset[1]

        if self.tensorflow_compatible:
            # to align with tensorflow
            # this is incorrect, I believe
            w += half_resolution[0]
            h += half_resolution[1]

        else:
            # directly do clamping
            w /= half_resolution[0]
            h /= half_resolution[1]

            # clamp to [-1, 1]
            w = torch.clamp(w, min=-1, max=1)
            h = torch.clamp(h, min=-1, max=1)

        feats = [inputs]
        for img_feature in img_features:
            feats.append(self.project(resolution, img_feature, torch.stack([w, h], dim=-1)))

        output = torch.cat(feats, 2)

        return output

    def project(self, img_shape, img_feat, sample_points):
        """
        :param img_shape: raw image shape
        :param img_feat: [batch_size x channel x h x w]
        :param sample_points: [batch_size x num_points x 2], in range [-1, 1]
        :return: [batch_size x num_points x feat_dim]
        """
        if self.tensorflow_compatible:
            feature_shape = self.image_feature_shape(img_feat)
            points_w = sample_points[:, :, 0] / (img_shape[0] / feature_shape[0])
            points_h = sample_points[:, :, 1] / (img_shape[1] / feature_shape[1])
            output = torch.stack([self.project_tensorflow(points_h[i], points_w[i],
                                                          feature_shape, img_feat[i]) for i in range(img_feat.size(0))], 0)
        else:
            output = F.grid_sample(img_feat, sample_points.unsqueeze(1))
            output = torch.transpose(output.squeeze(2), 1, 2)

        return output


class LocalGraphProjection(nn.Module):
    def __init__(self, cameras):
        super(LocalGraphProjection, self).__init__()

        #self.img_feat = placeholders['img_feat']
        self.camera = cameras


    def forward(self, inputs, img_feats, sample_views):
        coord = inputs
        out1_list = []
        out2_list = []
        out3_list = []
        out4_list = []
        out5_list = []
        out6_list = []

        point_origin = camera_trans_inv(self.camera['ext_rgb'][0], inputs)
        #n_view = img_feats[0].shape[0]
        # print(sample_views)
        for i in range(len(sample_views)):
            v = sample_views[i].numpy()[0]

            point_crrent = camera_trans(self.camera['ext_rgb'][v], point_origin)

            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2] + torch.tensor(1e-8)
            intr = self.camera['int_rgb'][v]

            orig_height = 1280.
            orig_width = 720.

            new_height = 448.
            new_width = 256.

            ratio_h = orig_height/new_height
            ratio_w = orig_width/new_width

            hupper = new_height -1.
            wupper = new_width -1.

            h = (float(intr['fy'])/ratio_h) * torch.divide(-Y, -Z) + float(intr['py'])/ratio_h
            w = (float(intr['fx'])/ratio_w) * torch.divide(X, -Z) + float(intr['px'])/ratio_w
            # print(torch.max(h), torch.min(h), torch.max(w), torch.min(w))


            h = torch.nan_to_num(h, nan=hupper, posinf=hupper)
            w = torch.nan_to_num(w, nan=wupper, posinf=wupper)

            h = torch.clamp(h, 0., hupper)
            w = torch.clamp(w, 0., wupper)
            #n = (torch.full(h.size(), i)).type(torch.FloatTensor).cuda()

            # x = (h / (224.0 / 224))
            # y = (w / (224.0 / 224))
            x = (h / (hupper / hupper))
            y = (w / (wupper / wupper))

            # x = torch.clamp(x, 0., 223.)
            # y = torch.clamp(y, 0., 223.)
            out1 = self.bi_linear_sample(img_feats[0][i], x, y, max_val=(hupper, wupper))
            out1_list.append(out1)

            # x = h / (224.0 / 112.)
            # y = w / (224.0 / 112.)
            x = h / 2.
            y = w / 2.
            # x = torch.clamp(x, 0., 111.)
            # y = torch.clamp(y, 0., 111.)
            out2 = self.bi_linear_sample(img_feats[1][i], x, y, max_val=(new_height/2.-1., new_width/2.-1.))
            out2_list.append(out2)

            x = h / 4.
            y = w / 4.
            # x = torch.clamp(x, 0., 55.)
            # y = torch.clamp(y, 0., 55.)
            out3 = self.bi_linear_sample(img_feats[2][i], x, y, max_val=(new_height/4.-1., new_width/4.-1.))
            out3_list.append(out3)

            x = h / 8.
            y = w / 8.
            # x = torch.clamp(x, 0., 27.)
            # y = torch.clamp(y, 0., 27.)
            out4 = self.bi_linear_sample(img_feats[3][i], x, y, max_val=(new_height/8.-1., new_width/8.-1.))
            out4_list.append(out4)

            del out1, out2, out3, out4
            torch.cuda.empty_cache()


            x = h / 16.
            y = w / 16.
            # x = torch.clamp(x, 0., 27.)
            # y = torch.clamp(y, 0., 27.)
            out5 = self.bi_linear_sample(img_feats[4][i], x, y, max_val=(new_height / 16. - 1., new_width / 16. - 1.))
            out5_list.append(out5)


            x = h / 32.
            y = w / 32.
            # x = torch.clamp(x, 0., 27.)
            # y = torch.clamp(y, 0., 27.)
            out6 = self.bi_linear_sample(img_feats[5][i], x, y, max_val=(new_height / 32. - 1., new_width / 32. - 1.))
            out6_list.append(out6)



        # ----

        all_out1 = torch.stack(out1_list, 0)
        all_out2 = torch.stack(out2_list, 0)
        all_out3 = torch.stack(out3_list, 0)
        all_out4 = torch.stack(out4_list, 0)
        all_out5 = torch.stack(out5_list, 0)
        all_out6 = torch.stack(out6_list, 0)

        # print(all_out1.shape)
        # print(all_out2.shape)
        # 3*N*[16+32+64] -> 3*N*F

        image_feature = torch.cat([all_out1, all_out2, all_out3, all_out4, all_out5, all_out6], 1) #,

        del out1_list, out2_list, out3_list, out4_list, all_out1, all_out2, all_out3#, all_out4
        torch.cuda.empty_cache()

        image_feature_max, _ = torch.max(image_feature, 0)
        image_feature_mean = torch.mean(image_feature, 0)
        image_feature_std = reduce_std(image_feature, 0)

        outputs = torch.cat([image_feature_max, image_feature_mean, image_feature_std], 0)
        #print(outputs.shape)

        return torch.transpose(outputs, 0, 1)

    def bi_linear_sample(self, img_feat, x, y, max_val):
        x1 = torch.clamp(torch.floor(x).long(), 0, max_val[0])
        x2 = torch.clamp(torch.ceil(x).long(), 0, max_val[0])
        y1 = torch.clamp(torch.floor(y).long(), 0, max_val[1])
        y2 = torch.clamp(torch.ceil(y).long(), 0, max_val[1])
        #n = n.type(torch.long)
        #print(x1.max(), x1.min(), x2.max(), x2.min(), y1.max(), y1.min(), y2.max(), y2.min())
        # Q11 = gather_nd(img_feat, torch.stack([n, x1.type(torch.int32), y1.type(torch.int32)], 1))
        #
        # Q12 = gather_nd(img_feat, torch.stack([n, x1.type(torch.int32), y2.type(torch.int32)], 1))
        #
        # Q21 = gather_nd(img_feat, torch.stack([n, x2.type(torch.int32), y1.type(torch.int32)], 1))
        #
        # Q22 = gather_nd(img_feat, torch.stack([n, x2.type(torch.int32), y1.type(torch.int32)], 1))

        #print('imgf', img_feat.shape)
        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        #print(Q11.shape, Q12.shape, Q22.shape, Q21.shape)

        weight1 = torch.mul(torch.subtract(x2.float(), x), torch.subtract(y2.float(), y))

        Q11 = torch.mul(torch.unsqueeze(weight1, 0), Q11)


        weight2 = torch.multiply(torch.subtract(x, x1.float()), torch.subtract(y2.float(), y))
        Q21 = torch.mul(torch.unsqueeze(weight2, 0), Q21)


        weight3 = torch.multiply(torch.subtract(x2.float(), x), torch.subtract(y, y1.float()))
        Q12 = torch.mul(torch.unsqueeze(weight3, 0), Q12)


        weight4 = torch.multiply(torch.subtract(x, x1.float()), torch.subtract(y, y1.float()))
        Q22 = torch.mul(torch.unsqueeze(weight4, 0), Q22)


        outputs = Q11 + Q21 + Q12 + Q22


        return outputs



class GraphProjection(nn.Module):
    def __init__(self, cameras, views):
        super(GraphProjection, self).__init__()

        #self.img_feat = placeholders['img_feat']
        self.camera = cameras
        self.view_number = views

    def forward(self, inputs, img_feats):
        coord = inputs
        out1_list = []
        out2_list = []
        out3_list = []
        out4_list = []

        for i in range(self.view_number):
            point_origin = camera_trans_inv(self.camera[0], inputs)
            point_crrent = camera_trans(self.camera[i], point_origin)
            X = point_crrent[:, 0]
            Y = point_crrent[:, 1]
            Z = point_crrent[:, 2]
            h = 248.0 * torch.divide(-Y, -Z) + 112.0
            w = 248.0 * torch.divide(X, -Z) + 112.0



            h = torch.clamp(h, 0, 223)
            w = torch.clamp(w, 0, 223)
            n = (torch.full(h.size(), i)).type(torch.FloatTensor)

            indeces = torch.stack([n, h, w], 1)

            idx = (indeces / (224.0 / 56.0)).type(torch.int32)
            out1 = th_gather_nd(img_feats[0], idx)
            out1_list.append(out1)
            idx = (indeces / (224.0 / 28.0)).type(torch.int32)
            out2 = th_gather_nd(img_feats[1], idx)
            out2_list.append(out2)
            idx = (indeces / (224.0 / 14.0)).type(torch.int32)
            out3 = th_gather_nd(img_feats[2], idx)
            out3_list.append(out3)
            idx = (indeces / (224.0 / 7.0)).type(torch.int32)
            out4 = th_gather_nd(img_feats[3], idx)
            out4_list.append(out4)
        # ----
        all_out1 = torch.stack(out1_list, 0)
        all_out2 = torch.stack(out2_list, 0)
        all_out3 = torch.stack(out3_list, 0)
        all_out4 = torch.stack(out4_list, 0)

        # 3*N*[64+128+256+512] -> 3*N*F
        image_feature = torch.cat([all_out1, all_out2, all_out3, all_out4], 2)
        # 3*N*F -> N*F
        # image_feature = tf.reshape(tf.transpose(image_feature, [1, 0, 2]), [-1, FLAGS.feat_dim * 3])

        #image_feature = tf.reduce_max(image_feature, axis=0)
        image_feature_max = torch.max(image_feature, 0)
        image_feature_mean = torch.mean(image_feature, 0)
        image_feature_std = reduce_std(image_feature, axis=0)

        outputs = torch.cat([coord, image_feature_max, image_feature_mean, image_feature_std], 1)
        return outputs


# class LocalGraphProjection(nn.Module):
#     def __init__(self, placeholders, views):
#         super(LocalGraphProjection, self).__init__()
#
#         self.img_feat = placeholders['img_feat']
#         self.camera = placeholders['cameras']
#         self.view_number = views
#
#     def _call(self, inputs):
#         coord = inputs
#         out1_list = []
#         out2_list = []
#         out3_list = []
#         # out4_list = []
#
#         for i in range(self.view_number):
#             point_origin = camera_trans_inv(self.camera[0], inputs)
#             point_crrent = camera_trans(self.camera[i], point_origin)
#             X = point_crrent[:, 0]
#             Y = point_crrent[:, 1]
#             Z = point_crrent[:, 2]
#             h = 248.0 * torch.divide(-Y, -Z) + 112.0
#             w = 248.0 * torch.divide(X, -Z) + 112.0
#
#             h = torch.minimum(torch.maximum(h, 0), 223)
#             w = tf.minimum(tf.maximum(w, 0), 223)
#             n = tf.cast(tf.fill(tf.shape(h), i), tf.int32)
#
#             x = h / (224.0 / 224)
#             y = w / (224.0 / 224)
#             out1 = self.bi_linear_sample(self.img_feat[0], n, x, y)
#             out1_list.append(out1)
#             x = h / (224.0 / 112)
#             y = w / (224.0 / 112)
#             out2 = self.bi_linear_sample(self.img_feat[1], n, x, y)
#             out2_list.append(out2)
#             x = h / (224.0 / 56)
#             y = w / (224.0 / 56)
#             out3 = self.bi_linear_sample(self.img_feat[2], n, x, y)
#             out3_list.append(out3)
#         # ----
#         all_out1 = tf.stack(out1_list, 0)
#         all_out2 = tf.stack(out2_list, 0)
#         all_out3 = tf.stack(out3_list, 0)
#
#         # 3*N*[16+32+64] -> 3*N*F
#         image_feature = tf.concat([all_out1, all_out2, all_out3], 2)
#
#         image_feature_max = tf.reduce_max(image_feature, axis=0)
#         image_feature_mean = tf.reduce_mean(image_feature, axis=0)
#         image_feature_std = reduce_std(image_feature, axis=0)
#
#         outputs = tf.concat([coord, image_feature_max, image_feature_mean, image_feature_std], 1)
#         return outputs
#
#     def bi_linear_sample(self, img_feat, n, x, y):
#         x1 = tf.floor(x)
#         x2 = tf.ceil(x)
#         y1 = tf.floor(y)
#         y2 = tf.ceil(y)
#         Q11 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 1))
#         Q12 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x1, tf.int32), tf.cast(y2, tf.int32)], 1))
#         Q21 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x2, tf.int32), tf.cast(y1, tf.int32)], 1))
#         Q22 = tf.gather_nd(img_feat, tf.stack([n, tf.cast(x2, tf.int32), tf.cast(y2, tf.int32)], 1))
#
#         weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y2, y))
#         Q11 = tf.multiply(tf.expand_dims(weights, 1), Q11)
#         weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y2, y))
#         Q21 = tf.multiply(tf.expand_dims(weights, 1), Q21)
#         weights = tf.multiply(tf.subtract(x2, x), tf.subtract(y, y1))
#         Q12 = tf.multiply(tf.expand_dims(weights, 1), Q12)
#         weights = tf.multiply(tf.subtract(x, x1), tf.subtract(y, y1))
#         Q22 = tf.multiply(tf.expand_dims(weights, 1), Q22)
#         outputs = tf.add_n([Q11, Q21, Q12, Q22])
#         return outputs
