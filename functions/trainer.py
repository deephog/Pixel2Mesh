import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mesh_load
from sknetwork.utils import edgelist2adjacency, bipartite2directed
import options
from functions.base import CheckpointRunner
#from functions.evaluator import Evaluator
from models.classifier import Classifier
from models.losses.classifier import CrossEntropyLoss
from models.losses.p2m_single import P2MLoss
from models.p2m import P2MHuman, MeshNet, P2MHumanGRU
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.tensor import recursive_detach
#from utils.vis.renderer import MeshRenderer
from multi_data import get_loader
from torch_geometric import utils
import collections
import os


class Trainer(CheckpointRunner):

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        if self.options.model.name == "pixel2mesh":
            #self.renderer = None
            # Visualization renderer
            # self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
            #                              self.options.dataset.mesh_pos)
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        else:
            self.renderer = None

        if shared_model is not None:
            self.model = shared_model
            self.model.cuda()
        else:
            if self.options.model.name == "pixel2mesh":
                # create model
                cameras = mesh_load.camera_setting_loader(os.path.join(self.options.dataset.data_dir, 'default_camera_setting.xml'))
                self.model = P2MHuman(self.options.model, cameras=cameras)
            elif self.options.model.name == "classifier":
                self.model = Classifier(self.options.model, self.options.dataset.num_classes)
            else:
                raise NotImplementedError("Your model is not found")
            #self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()
            self.model.cuda()
        print(self.model)

        # Setup a joint optimizer for the 2 models
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif self.options.optim.name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        # Create loss functions
        if self.options.model.name == "pixel2mesh":
            self.criterion = P2MLoss(self.options.loss).cuda()
        elif self.options.model.name == "classifier":
            self.criterion = CrossEntropyLoss()
        else:
            raise NotImplementedError("Your loss is not found")

        # Create AverageMeters for losses
        self.losses = AverageMeter()

        # Evaluators

        self.train_data_loader = get_loader(data_root=self.options.dataset.data_dir,
                                       mesh_root='/home/hypevr/nas/2021/09/0927/deform_data_set_02/LowRes/mesh_with_normal/',
                                       lap_root='/home/hypevr/nas/2021/09/0927/deform_data_set_02/LowRes/laplacian/',
                                       batchsize=1,
                                       trainsize=(256, 448),
                                       origsize=None,
                                       n_views=8,
                                       shuffle=True,
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
        #self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]

    def models_dict(self):
        return {'model': self.model}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        self.model.train()

        #image, this_mesh, next_mesh, names, sample_views, mask, depth = input_batch
        seq = input_batch

        num_steps = len(seq)

        # print(names[0])
        # Grab data from the batch
        #images = torch.squeeze(image, dim=0).cuda()
        images = []
        sample_views = []
        for s in seq:
            images.append(torch.squeeze(s['depth'], dim=0).cuda())
            sample_views.append(s['sample_views'])
        verts = torch.squeeze(seq[0]['this_mesh']['norm_vert']).cuda()

        #cent = torch.squeeze(this_mesh['cent']).cuda()
        #diag_len = torch.squeeze(this_mesh['diag']).cuda()
        verts = torch.transpose(verts, 0, 1)
        edges = torch.squeeze(seq[0]['this_mesh']['edge']).cuda()
        #adj = edgelist2adjacency(edges.numpy())

        matrix = utils.to_dense_adj(edges)

        #print(this_mesh)
        #this_mesh = torch.squeeze(this_mesh).cuda()

        # predict with model
        out = self.model(images, verts, matrix, sample_views, recover=[None, 3000.])

        this_mesh = seq[0]['this_mesh']
        # compute loss
        total_loss = torch.tensor([0.]).cuda()
        loss_sum = {
            "loss": torch.tensor([0.]).cuda(),
            "loss_chamfer": torch.tensor([0.]).cuda(),
            "loss_edge": torch.tensor([0.]).cuda(),
            "loss_laplace": torch.tensor([0.]).cuda(),
            "loss_move": torch.tensor([0.]).cuda(),
            "loss_normal": torch.tensor([0.]).cuda(),
        }
        for i in range(num_steps):
            loss, loss_summary = self.criterion(out[i], this_mesh, seq[i]['next_mesh'], images[i])
            total_loss += loss
            a_counter = collections.Counter(loss_summary)
            b_counter = collections.Counter(loss_sum)

            add_dict = a_counter + b_counter
            loss_sum = dict(add_dict)

        loss = total_loss
        self.losses.update(loss.detach().cpu().item())

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def train(self):
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            # Create a new data loader for every epoch
            # train_data_loader = DataLoader(self.dataset,
            #                                batch_size=self.options.train.batch_size * self.options.num_gpus,
            #                                num_workers=self.options.num_workers,
            #                                pin_memory=self.options.pin_memory,
            #                                shuffle=self.options.train.shuffle,
            #                                collate_fn=self.dataset_collate_fn)


            # Reset loss
            self.losses.reset()

            # Iterate over all batches in an epoch
            for step, batch in enumerate(self.train_data_loader):
                # Send input to GPU
                #batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                #image, this_mesh, next_mesh, sample_views = batch
                # Run training step
                out = self.train_step(batch)

                self.step_count += 1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()

            # save checkpoint after each epoch
            self.dump_checkpoint()

            # Run validation every test_epochs
            # if self.epoch_count % self.options.train.test_epochs == 0:
            #     self.test()

            # lr scheduler step
            self.lr_scheduler.step()

    def train_summaries(self, input_batch, out_summary, loss_summary):
        # if self.renderer is not None:
        #     # Do visualization for the first 2 images of the batch
        #     render_mesh = self.renderer.p2m_batch_visualize(input_batch, out_summary, self.ellipsoid.faces)
        #     self.summary_writer.add_image("render_mesh", render_mesh, self.step_count)
        #     self.summary_writer.add_histogram("length_distribution", input_batch["length"].cpu().numpy(),
        #                                       self.step_count)

        # Debug info for filenames
        #self.logger.debug(input_batch["filename"])

        # Save results in Tensorboard
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

        # Save results to log
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.9f (%.9f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * 1342 // (
                        self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))

    def test(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()
