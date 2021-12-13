import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck
from models.layers.gconv import GConv, GConvGated
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection, GraphProjection, LocalGraphProjection
from models.layers.mods import *
from models.BASNet import Bone34


class P2MModel(nn.Module):

    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super(P2MModel, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[0], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.coord_dim,
                        ellipsoid.adj_mat[1], activation=self.gconv_activation),
            GBottleneck(6, self.features_dim + self.hidden_dim, self.hidden_dim, self.last_hidden_dim,
                        ellipsoid.adj_mat[2], activation=self.gconv_activation)
        ])

        self.unpooling = nn.ModuleList([
            GUnpooling(ellipsoid.unpool_idx[0]),
            GUnpooling(ellipsoid.unpool_idx[1])
        ])

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
                           adj_mat=ellipsoid.adj_mat[2])

    def forward(self, img):
        batch_size = img.size(0)
        img_feats = self.nn_encoder(img)
        img_shape = self.projection.image_feature_shape(img)

        init_pts = self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
        # GCN Block 1
        x = self.projection(img_shape, img_feats, init_pts)
        x1, x_hidden = self.gcns[0](x)

        # before deformation 2
        x1_up = self.unpooling[0](x1)

        # GCN Block 2
        x = self.projection(img_shape, img_feats, x1)
        x = self.unpooling[0](torch.cat([x, x_hidden], 2))
        # after deformation 2
        x2, x_hidden = self.gcns[1](x)

        # before deformation 3
        x2_up = self.unpooling[1](x2)

        # GCN Block 3
        x = self.projection(img_shape, img_feats, x2)
        x = self.unpooling[1](torch.cat([x, x_hidden], 2))
        x3, _ = self.gcns[2](x)
        if self.gconv_activation:
            x3 = F.relu(x3)
        # after deformation 3
        x3 = self.gconv(x3)

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_feats)
        else:
            reconst = None

        return {
            "pred_coord": [x1, x2, x3],
            "pred_coord_before_deform": [init_pts, x1_up, x2_up],
            "reconst": reconst
        }




class P2MHuman(nn.Module):

    def __init__(self, options, cameras):
        super(P2MHuman, self).__init__()
        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        #self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.nn_decoder = None
        #self.nn_encoder = Bone34(3)
        self.features_dim = 3030#self.nn_encoder.features_dim * 4 + self.coord_dim
        # print(self.features_dim)

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
                        activation=self.gconv_activation),
            # GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
            #             activation=self.gconv_activation),
            GBottleneck(6, self.features_dim, self.hidden_dim, self.last_hidden_dim,
                        activation=self.gconv_activation)
        ])

        # self.unpooling = nn.ModuleList([
        #     GUnpooling(ellipsoid.unpool_idx[0]),
        #     GUnpooling(ellipsoid.unpool_idx[1])
        # ])

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = LocalGraphProjection(cameras)

        self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim)

    def forward(self, images, input_verts, adj, input_views, recover):
        #batch_size = img.size(0)
        #img_feats = []
        # for i in self.views:
        #     img_feats.append(self.nn_encoder(img[i]))
        num_steps = len(images)
        verts = input_verts
        return_list = []

        for n in range(num_steps):
            img = images[n]
            views = input_views[n]
            img_feats = self.nn_encoder(img)
            #print(img_feats[2].shape)

            #img_shape = self.projection.image_feature_shape(img)

            # init_pts = last_mesh#self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
            # GCN Block 1

            x = self.projection(verts*recover[1], img_feats, views)
            x1 = torch.cat([torch.transpose(verts, 0, 1), x], dim=1)
            x1, x_hidden = self.gcns[0](x1, adj)
            x1 += torch.transpose(verts, 0, 1)

            # # before deformation 2
            # x1_up = self.unpooling[0](x1)
            # GCN Block 2
            x = self.projection(torch.transpose(x1, 0, 1)*recover[1], img_feats, views)
            # x = self.unpooling[0](torch.cat([x, x_hidden], 2))
            # after deformation 2
            x2 = torch.cat([x1, x], dim=1)
            x2, x_hidden = self.gcns[1](x2, adj)
            #x2 += x1

            # x = self.projection(torch.transpose(x2, 0, 1) * recover[1], img_feats, views)
            # # x = self.unpooling[0](torch.cat([x, x_hidden], 2))
            # # after deformation 2
            # x3 = torch.cat([x2, x], dim=1)
            # x3, x_hidden = self.gcns[2](x3, adj)

            # # before deformation 3
            # x2_up = self.unpooling[1](x2)

            # # GCN Block 3
            # x = self.projection(torch.transpose(x2, 0, 1), img_feats)
            # # x = self.unpooling[1](torch.cat([x, x_hidden], 2))
            # x3, _ = self.gcns[2](x, adj)
            # if self.gconv_activation:
            #     x3 = F.relu(x3)
            # # after deformation 3
            x2 = self.gconv(x2, adj)
            x2 += x1
            #print(img_feats.shape)
            if self.nn_decoder is not None:
                reconst = self.nn_decoder(img_feats)
            else:
                reconst = None

            return_list.append({
            "pred_coord": [x1*recover[1], x2*recover[1]],#, x3*recover[1]],
            "pred_coord_before_deform": verts*recover[1],
            "reconst": reconst
            })
            verts = torch.transpose(x2, 0, 1)

        return return_list


class P2MHumanGRU(nn.Module):

    def __init__(self, options, cameras):
        super(P2MHumanGRU, self).__init__()
        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        #self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.nn_decoder = None
        #self.nn_encoder = Bone34(3)
        self.features_dim = 3030#self.nn_encoder.features_dim * 4 + self.coord_dim
        # print(self.features_dim)

        self.gcns = nn.ModuleList([
            GBottleneck(6, self.features_dim, self.hidden_dim, self.last_hidden_dim,
                        activation=self.gconv_activation),
            # GBottleneck(6, self.features_dim, self.hidden_dim, self.coord_dim,
            #             activation=self.gconv_activation),
            GBottleneck(6, self.features_dim, self.hidden_dim, self.last_hidden_dim,
                        activation=self.gconv_activation)
        ])

        self.gru1 = GConvGated(self.hidden_dim, self.hidden_dim)#nn.GRUCell(self.hidden_dim, self.hidden_dim)##nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.gru2 = GConvGated(self.hidden_dim, self.hidden_dim)#nn.GRUCell(self.hidden_dim, self.hidden_dim)#GConvGated(self.hidden_dim, self.hidden_dim)

        # self.unpooling = nn.ModuleList([
        #     GUnpooling(ellipsoid.unpool_idx[0]),
        #     GUnpooling(ellipsoid.unpool_idx[1])
        # ])

        # if options.align_with_tensorflow:
        #     self.projection = GProjection
        # else:
        #     self.projection = GProjection
        self.projection = LocalGraphProjection(cameras)

        self.gconv1 = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim)
        self.gconv2 = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim)

    def forward(self, images, input_verts, adj, input_views, recover):
        #batch_size = img.size(0)
        #img_feats = []
        # for i in self.views:
        #     img_feats.append(self.nn_encoder(img[i]))
        num_steps = len(images)
        verts = input_verts
        return_list = []

        num_verts = verts.shape[1]
        h1 = torch.zeros([num_verts, self.last_hidden_dim]).cuda()
        h2 = torch.zeros([num_verts, self.last_hidden_dim]).cuda()

        for n in range(num_steps):
            img = images[n]
            views = input_views[n]
            img_feats = self.nn_encoder(img)
            #print(img_feats[2].shape)

            #img_shape = self.projection.image_feature_shape(img)

            # init_pts = last_mesh#self.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)
            # GCN Block 1

            x = self.projection(verts*recover[1], img_feats, views)
            x1 = torch.cat([torch.transpose(verts, 0, 1), x], dim=1)
            x1, x_hidden = self.gcns[0](x1, adj)
            h1 = self.gru1(x1, h1, adj)
            x1 = self.gconv1(h1, adj)
            x1 += torch.transpose(verts, 0, 1)

            # # before deformation 2
            # x1_up = self.unpooling[0](x1)
            # GCN Block 2
            x = self.projection(torch.transpose(x1, 0, 1)*recover[1], img_feats, views)
            # x = self.unpooling[0](torch.cat([x, x_hidden], 2))
            # after deformation 2
            x2 = torch.cat([x1, x], dim=1)
            x2, x_hidden = self.gcns[1](x2, adj)
            h2 = self.gru2(x2, h2, adj)
            x2 = self.gconv2(h2, adj)
            x2 += x1
            #x2 += x1

            # x = self.projection(torch.transpose(x2, 0, 1) * recover[1], img_feats, views)
            # # x = self.unpooling[0](torch.cat([x, x_hidden], 2))
            # # after deformation 2
            # x3 = torch.cat([x2, x], dim=1)
            # x3, x_hidden = self.gcns[2](x3, adj)

            # # before deformation 3
            # x2_up = self.unpooling[1](x2)

            # # GCN Block 3
            # x = self.projection(torch.transpose(x2, 0, 1), img_feats)
            # # x = self.unpooling[1](torch.cat([x, x_hidden], 2))
            # x3, _ = self.gcns[2](x, adj)
            # if self.gconv_activation:
            #     x3 = F.relu(x3)
            # # after deformation 3

            #print(img_feats.shape)
            if self.nn_decoder is not None:
                reconst = self.nn_decoder(img_feats)
            else:
                reconst = None

            return_list.append({
            "pred_coord": [x1*recover[1], x2*recover[1]],#, x3*recover[1]],
            "pred_coord_before_deform": verts*recover[1],
            "reconst": reconst
            })
            verts = torch.transpose(x2, 0, 1)

        return return_list



class MeshNet(nn.Module):
    def __init__(self, options, cameras):
        super(MeshNet, self).__init__()

        self.options = options
        self.cameras = cameras

        self.nn_encoder, self.nn_decoder = get_backbone(options)
        self.nn_decoder = None
        #self.build_cnn18()  # update image feature
        # sample hypothesis points

        self.sample1 = SampleHypothesis(options=self.options)
        # 1st projection block
        self.proj1 = LocalGraphProjection(cameras)
        # 1st DRB
        self.drb1 = DeformationReasoning(input_dim=129,
                                         output_dim=3,
                                         options=self.options,
                                         gcn_block=3
                                         )
        # sample hypothesis points
        self.sample2 = SampleHypothesis(options=self.options)
        # 2nd projection block
        self.proj2 = LocalGraphProjection(cameras=cameras)
        # 2nd DRB
        self.drb2 = DeformationReasoning(input_dim=129,
                                         output_dim=3,
                                         options=self.options,
                                         gcn_block=3)
        # self.dense1 = nn.Linear(7168, 192)
        # self.dense2 = nn.Linear(192, 192)
        # self.dense3 = nn.Linear(192, 2)
        # self.dense1 = nn.Linear(129, 192)
        # self.dense2 = nn.Linear(192, 1)
        # self.relu = nn.LeakyReLU()

        # self.sample3 = SampleHypothesis(options=self.options)
        # # 2nd projection block
        # self.proj3 = LocalGraphProjection(cameras=cameras)
        # # 2nd DRB
        # self.drb3 = DeformationReasoning(input_dim=339,
        #                                  output_dim=3,
        #                                  options=self.options,
        #                                  gcn_block=3)
        #
        # self.sample4 = SampleHypothesis(options=self.options)
        # # 2nd projection block
        # self.proj4 = LocalGraphProjection(cameras=cameras)
        # # 2nd DRB
        # self.drb4 = DeformationReasoning(input_dim=339,
        #                                  output_dim=3,
        #                                  options=self.options,
        #                                  gcn_block=3)

        self.const = Variable(torch.tensor([1.], requires_grad=False).type(torch.float32).cuda())


    def forward(self, img_feats, verts, views, recover):
        ''' Wrapper for _build() '''
        #img_feats = self.nn_encoder(img)

        # mag_const = torch.mean(img_feats[-1], dim=(0, 1)).view(-1)
        # mag_const = self.dense3(self.dense2(self.dense1(mag_c


        blk1_sample = self.sample1(verts, self.const)
        blk1_sample_rec = blk1_sample * recover[1]# + recover[0]
        blk1_sample_rec = torch.transpose(blk1_sample_rec, 0, 1)
        blk1_proj_feat = self.proj1(blk1_sample_rec, img_feats, views)
        #print(blk1_sample.shape, blk1_proj_feat.shape)
        blk1_proj_feat = torch.cat([blk1_sample, blk1_proj_feat], dim=1)
        #print(blk1_proj_feat.shape)
        blk1_out = self.drb1((blk1_proj_feat, verts))

        # const_adj = torch.mean(blk1_proj_feat, dim=0)
        # const_adj = self.dense2(self.dense1(const_adj)) + 100.
        #print(const_adj)
        # self.const += const_adj


        blk2_sample = self.sample2(blk1_out, self.const)
        blk2_sample_rec = blk2_sample * recover[1]# + recover[0]
        blk2_sample_rec = torch.transpose(blk2_sample_rec, 0, 1)
        blk2_proj_feat = self.proj2(blk2_sample_rec, img_feats, views)

        blk2_proj_feat = torch.cat([blk2_sample, blk2_proj_feat], dim=1)
        blk2_out = self.drb2((blk2_proj_feat, blk1_out))

        #blk3_sample = self.sample3(blk2_out, self.const)
        # blk3_sample = torch.transpose(blk2_out, 0, 1)
        # blk3_proj_feat = self.proj3(blk3_sample, img_feats, views)
        # blk3_out = self.drb3((blk3_proj_feat, blk2_out))
        #
        # #blk4_sample = self.sample4(blk3_out, self.const)
        # blk4_sample = torch.transpose(blk3_out, 0, 1)
        # blk4_proj_feat = self.proj4(blk4_sample, img_feats, views)
        # blk4_out = self.drb4((blk4_proj_feat, blk3_out))

        # const_adj2 = torch.mean(blk2_proj_feat, dim=0)
        # const_adj2 = self.dense2(self.dense1(const_adj2))
        # self.const += const_adj2
        # self.output1l = blk1_out
        # self.output2l = blk2_out

        # Store model variables for easy access
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name+'/')
        # self.vars = {var.name: var for var in variables}
        #
        # # Build metrics
        # self._loss()
        # self.opt_op = self.optimizer.minimize(self.loss)
        #
        # self.summary_loss = tf.summary.scalar('loss', self.loss)
        # self.merged_summary_op = tf.summary.merge_all()

        blk1_out = blk1_out * recover[1]# + recover[0]
        blk2_out = blk2_out * recover[1]# + recover[0]

        if self.nn_decoder is not None:
            reconst = self.nn_decoder(img_feats)
        else:
            reconst = None
        return {
            "pred_coord": [blk1_out, blk2_out],
            "pred_coord_before_deform": verts,
            "reconst": reconst
        }