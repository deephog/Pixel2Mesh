import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .resnet_model import *


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


class BASNet101(nn.Module):
    def __init__(self, n_channels, n_classes, test_mode=False):
        super(BASNet101, self).__init__()
        self.test_mode = test_mode
        resnet = models.resnet101(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)
        # self.downsample = nn.MaxPool2d(2,2,ceil_mode=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.resb5_0 = torch.nn.Conv2d(2048, 512, 1, 1, padding=0)
        self.resb4_0 = torch.nn.Conv2d(1024, 256, 1, 1, padding=0)
        self.resb3_0 = torch.nn.Conv2d(512, 128, 1, 1, padding=0)
        self.resb2_0 = torch.nn.Conv2d(256, 64, 1, 1, padding=0)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7

        ## -------------Bridge--------------

        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  ###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, n_classes, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)

    def forward(self, x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)
        # hx = self.downsample(hx)

        h1 = self.encoder1(hx)  # 256
        h1x = self.resb2_0(h1)
        h2 = self.encoder2(h1)  # 128
        h2x = self.resb3_0(h2)
        h3 = self.encoder3(h2)  # 64
        h3x = self.resb4_0(h3)
        h4 = self.encoder4(h3)  # 32
        h4 = self.resb5_0(h4)

        hx = self.pool4(h4)  # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5)  # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))  # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6)  # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5)  # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)  # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3x), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)  # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2x), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1x), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        if not self.test_mode:
            db = self.outconvb(hbg)
            db = self.upscore6(db)  # 8->256

            d6 = self.outconv6(hd6)
            d6 = self.upscore6(d6)  # 8->256

            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5)  # 16->256

            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4)  # 32->256

            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3)  # 64->256

            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2)  # 128->256

            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1) # 256
            #dout = d1

            return dout.sigmoid(), d1.sigmoid(), d2.sigmoid(), d3.sigmoid(), d4.sigmoid(), d5.sigmoid(), d6.sigmoid(), db.sigmoid()
        else:
            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1)  # 256
            #dout = d1

            return dout.sigmoid()


class BASNet50(nn.Module):
    def __init__(self, n_channels, n_classes, test_mode=False):
        super(BASNet50, self).__init__()
        self.test_mode = test_mode
        resnet = models.resnet50(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)
        # self.downsample = nn.MaxPool2d(2,2,ceil_mode=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.resb5_0 = torch.nn.Conv2d(2048, 512, 1, 1, padding=0)
        self.resb4_0 = torch.nn.Conv2d(1024, 256, 1, 1, padding=0)
        self.resb3_0 = torch.nn.Conv2d(512, 128, 1, 1, padding=0)
        self.resb2_0 = torch.nn.Conv2d(256, 64, 1, 1, padding=0)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7

        ## -------------Bridge--------------

        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  ###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, n_classes, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)

    def forward(self, x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)
        # hx = self.downsample(hx)

        h1 = self.encoder1(hx)  # 256
        h1x = self.resb2_0(h1)
        h2 = self.encoder2(h1)  # 128
        h2x = self.resb3_0(h2)
        h3 = self.encoder3(h2)  # 64
        h3x = self.resb4_0(h3)
        h4 = self.encoder4(h3)  # 32
        h4 = self.resb5_0(h4)

        hx = self.pool4(h4)  # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5)  # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))  # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6)  # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5)  # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)  # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3x), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)  # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2x), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1x), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        if not self.test_mode:
            db = self.outconvb(hbg)
            db = self.upscore6(db)  # 8->256

            d6 = self.outconv6(hd6)
            d6 = self.upscore6(d6)  # 8->256

            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5)  # 16->256

            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4)  # 32->256

            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3)  # 64->256

            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2)  # 128->256

            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1) # 256
            #dout = d1

            return dout.sigmoid(), d1.sigmoid(), d2.sigmoid(), d3.sigmoid(), d4.sigmoid(), d5.sigmoid(), d6.sigmoid(), db.sigmoid()
        else:
            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1)  # 256
            #dout = d1

            return dout.sigmoid()



class Bone34(nn.Module):
    def __init__(self, n_channels, test_mode=False):
        super(Bone34, self).__init__()
        self.test_mode = test_mode
        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)
        # self.downsample = nn.MaxPool2d(2,2,ceil_mode=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28



    def forward(self, x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)
        # hx = self.downsample(hx)

        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32


        return [h1, h2, h3, h4]






class BASNet(nn.Module):
    def __init__(self, n_channels, n_classes, test_mode=False):
        super(BASNet, self).__init__()
        self.test_mode = test_mode
        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)
        # self.downsample = nn.MaxPool2d(2,2,ceil_mode=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56
        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7

        ## -------------Bridge--------------

        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  ###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, n_classes, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)

    def forward(self, x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)
        # hx = self.downsample(hx)

        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32

        hx = self.pool4(h4)  # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5)  # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))  # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6)  # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5)  # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)  # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)  # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        if not self.test_mode:
            db = self.outconvb(hbg)
            db = self.upscore6(db)  # 8->256

            d6 = self.outconv6(hd6)
            d6 = self.upscore6(d6)  # 8->256

            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5)  # 16->256

            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4)  # 32->256

            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3)  # 64->256

            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2)  # 128->256

            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1) # 256
            #dout = d1

            return dout.sigmoid(), d1.sigmoid(), d2.sigmoid(), d3.sigmoid(), d4.sigmoid(), d5.sigmoid(), d6.sigmoid(), db.sigmoid()
        else:
            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1)  # 256
            #dout = d1

            return dout.sigmoid()


class BASNet18(nn.Module):
    def __init__(self, n_channels, n_classes, test_mode=False):
        super(BASNet18, self).__init__()
        self.test_mode = test_mode
        resnet = models.resnet18(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 224
        # stage 2
        self.encoder2 = resnet.layer2  # 112
        # stage 3
        self.encoder3 = resnet.layer3  # 56

        # stage 4
        self.encoder4 = resnet.layer4  # 28

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        self.resb5_1 = BasicBlock(512, 512)
        self.resb5_2 = BasicBlock(512, 512)
        self.resb5_3 = BasicBlock(512, 512)  # 14
        self.rfb5_4 = RFB(512, 512)

        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 6
        self.resb6_1 = BasicBlock(512, 512)
        self.resb6_2 = BasicBlock(512, 512)
        self.resb6_3 = BasicBlock(512, 512)  # 7
        self.rfb6_4 = RFB(512, 512)

        ## -------------Bridge--------------

        # stage Bridge
        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d
        self.conv6d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  ###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1, 64)

    def forward(self, x):

        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        h1 = self.encoder1(hx)  # 256
        h2 = self.encoder2(h1)  # 128
        h3 = self.encoder3(h2)  # 64
        h4 = self.encoder4(h3)  # 32

        hx = self.pool4(h4)  # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)
        h5 = self.rfb5_4(h5)

        hx = self.pool5(h5)  # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)
        h6 = self.rfb6_4(h6)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6)))  # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg, h6), 1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6)  # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5)  # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)  # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)  # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        if not self.test_mode:
            db = self.outconvb(hbg)
            db = self.upscore6(db)  # 8->256

            d6 = self.outconv6(hd6)
            d6 = self.upscore6(d6)  # 8->256

            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5)  # 16->256

            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4)  # 32->256

            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3)  # 64->256

            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2)  # 128->256

            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1)  # 256

            return dout.sigmoid(), d1.sigmoid(), d2.sigmoid(), d3.sigmoid(), d4.sigmoid(), d5.sigmoid(), d6.sigmoid(), db.sigmoid()
        else:
            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            dout = self.refunet(d1)  # 256

            return dout.sigmoid()


class BASNet34_RFB(nn.Module):
    def __init__(self, n_channels, n_classes, test_mode=False):
        super(BASNet34_RFB, self).__init__()
        self.test_mode = test_mode
        resnet = models.resnet34(pretrained=True)

        ## -------------Encoder--------------

        self.inconv = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.inbn = nn.BatchNorm2d(64)
        self.inrelu = nn.ReLU(inplace=True)

        ##self.resb1_1 = BasicBlock(64, 64)
        #self.resb1_2 = BasicBlock(64, 64)
        #self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 1
        self.encoder1 = resnet.layer1  # 256
        # stage 2
        self.encoder2 = resnet.layer2  # 128
        # stage 3
        self.encoder3 = resnet.layer3  # 64
        # stage 4
        self.encoder4 = resnet.layer4  # 32

        #self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # stage 5
        # self.resb5_1 = BasicBlock(512,512)
        # self.resb5_2 = BasicBlock(512,512)
        # self.resb5_3 = BasicBlock(512,512) #16
        self.resb = RFB(512, 512)
        # self.resb5_2 = RFB(512, 512)
        # self.resb5_3 = RFB(512, 512)  # 16
        # self.resb5_4 = RFB(512, 512)

        # self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        # stage 6
        # self.resb6_1 = RFB(256, 256)
        # self.resb6_2 = RFB(256, 256)
        # self.resb6_3 = RFB(256, 256) #8

        ## -------------Bridge--------------

        # stage Bridge
        bridge_size = 512
        self.convbg_1 = nn.Conv2d(bridge_size, bridge_size, 3, dilation=2, padding=2)  # 7
        self.bnbg_1 = nn.BatchNorm2d(bridge_size)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(bridge_size, bridge_size, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(bridge_size)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(bridge_size, bridge_size, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(bridge_size)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        # stage 6d
        # self.conv6d_1 = nn.Conv2d(512,256,3,padding=1) # 16
        # self.bn6d_1 = nn.BatchNorm2d(256)
        # self.relu6d_1 = nn.ReLU(inplace=True)
        #
        # self.conv6d_m = nn.Conv2d(256,256,3,dilation=2, padding=2)###
        # self.bn6d_m = nn.BatchNorm2d(256)
        # self.relu6d_m = nn.ReLU(inplace=True)
        #
        # self.conv6d_2 = nn.Conv2d(256,256,3,dilation=2, padding=2)
        # self.bn6d_2 = nn.BatchNorm2d(256)
        # self.relu6d_2 = nn.ReLU(inplace=True)

        # stage 5d
        # self.conv5d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 16
        # self.bn5d_1 = nn.BatchNorm2d(512)
        # self.relu5d_1 = nn.ReLU(inplace=True)
        #
        # self.conv5d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        # self.bn5d_m = nn.BatchNorm2d(512)
        # self.relu5d_m = nn.ReLU(inplace=True)
        #
        # self.conv5d_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.bn5d_2 = nn.BatchNorm2d(512)
        # self.relu5d_2 = nn.ReLU(inplace=True)

        # stage 4d
        self.conv4d_1 = nn.Conv2d(1024, 512, 3, padding=1)  # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512, 512, 3, padding=1)  ###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        # stage 3d
        self.conv3d_1 = nn.Conv2d(512, 256, 3, padding=1)  # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256, 256, 3, padding=1)  ###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        # stage 2d

        self.conv2d_1 = nn.Conv2d(256, 128, 3, padding=1)  # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128, 128, 3, padding=1)  ###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        # stage 1d
        self.conv1d_1 = nn.Conv2d(128, 64, 3, padding=1)  # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64, 64, 3, padding=1)  ###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32, mode='bilinear')  ###
        #self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512, 1, 3, padding=1)
        #self.outconv6 = nn.Conv2d(512, 1, 3, padding=1)
        #self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        ## -------------Refine Module-------------
        #self.refunet = RefUnet(1, 64)
        self.convfilter = ConvGuidedFilter(views=1)

    def forward(self, x_orig):
        x = F.interpolate(x_orig, size=(256, 256), mode='bilinear', align_corners=True)
        hx = x

        ## -------------Encoder-------------
        hx = self.inconv(hx)
        hx = self.inbn(hx)
        hx = self.inrelu(hx)

        #h0 = self.resb1_1(hx)
        # h0 = self.resb1_2(h0)
        #hx = self.pool1(h0)

        h1 = self.encoder1(hx)  # 128
        h2 = self.encoder2(h1)  # 64
        h3 = self.encoder3(h2)  # 32
        h4 = self.encoder4(h3)  # 16

        #hx = self.pool4(h4)  # 16

        #hx = self.resb5_1(hx)
        #hx = self.resb5_2(hx)
        h5 = self.resb(h4)
        # h5 = self.resb5_4(hx)

        # hx = self.pool5(h5) # 8

        # hx = self.resb6_1(hx)
        # h6 = self.resb6_2(hx)
        # h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h5)))  # 8
        #hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        # hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h5),1))))
        # hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        # hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        # hx = self.upscore2(hd6) # 8 -> 16

        # hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx, h5), 1))))
        # hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        # hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))
        #
        # hx = self.upscore2(hd5)  # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hbg, h4), 1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4)  # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx, h3), 1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3)  # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx, h2), 1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2)  # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx, h1), 1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        if not self.test_mode:
            #db = self.outconvb(h5)
            #db = self.upscore4(db)  # 8->256

            # d6 = self.outconv6(hd6)
            # d6 = self.upscore6(d6) # 8->256

            #d5 = self.outconv5(hd5)
            #d5 = self.upscore5(d5)  # 16->256

            #d4 = self.outconv4(hd4)
            #d4 = self.upscore4(d4)  # 32->256

            #d3 = self.outconv3(hd3)
            #d3 = self.upscore3(d3)  # 64->256

            #d2 = self.outconv2(hd2)
            #d2 = self.upscore2(d2)  # 128->256

            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            #dout = self.refunet(d1)  # 256
            dout = self.convfilter(x, d1, x_orig)
            return dout.sigmoid(), d1.sigmoid()#, d2.sigmoid()#, d3.sigmoid()#, d4.sigmoid(), db.sigmoid()#, d5.sigmoid()
        else:
            d1 = self.outconv1(hd1)  # 256

            ## -------------Refine Module-------------
            #dout = self.refunet(d1)  # 256
            dout = self.convfilter(x, d1, x_orig)

            return dout.sigmoid()


class ConvGuidedFilter(nn.Module):
    def __init__(self, views, radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False)  # , groups=3)
        self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 3, kernel_size=1, bias=False))
        self.box_filter.weight.data.fill_(1.0)
        self.compress = nn.Conv2d(3, 1, 1, padding=0)
        self.views = views
        #self.upper = nn.Upsample(size=(1280, 720), mode='bilinear', align_corners=True)
        # self.upscore = nn.Upsample(size=8, mode='bilinear')

    def forward(self, x_lr, y_lr, x_hr):
        chs = self.views
        ots = []
        for i in range(chs):
            ot = self.forward_single(x_lr[:, (3 * i):(3 * i + 3), :, :], y_lr[:, i:i + 1, :, :],
                                     x_hr[:, (3 * i):(3 * i + 3), :, :])
            ots.append(ot)
        ot = torch.cat(ots, dim=1)
        return ot

    def forward_single(self, x_lr, y_lr, x_hr):
        # _, _, h_lrx, w_lrx = x_lr.size()
        # _, _, h_hrx, w_hrx = x_hr.size()

        y_lr = torch.cat([y_lr, y_lr, y_lr], dim=1)

        out_size = (x_hr.shape[2], x_hr.shape[3])

        # N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
        N = self.box_filter(
            torch.ones((x_lr.size(0), x_lr.size(1), x_lr.size(2), x_lr.size(3)), dtype=x_lr.dtype, device=x_lr.device))
        ## mean_x
        mean_x = self.box_filter(x_lr) / N
        ## mean_y
        mean_y = self.box_filter(y_lr) / N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        ## var_x
        var_x = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, size=out_size, mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, size=out_size, mode='bilinear', align_corners=True)
        #print(mean_A.shape, mean_b.shape, x_hr.shape)

        return self.compress(mean_A * x_hr + mean_b)
