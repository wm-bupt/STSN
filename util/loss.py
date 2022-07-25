import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGGLoss_for_trans(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss_for_trans, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, trans_img, struct_img, texture_img, weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]):
        while trans_img.size()[3] > 1024:
            trans_img, struct_img, texture_img = self.downsample(trans_img), self.downsample(struct_img), self.downsample(texture_img)
        trans_vgg, struct_vgg, texture_vgg = self.vgg(trans_img), self.vgg(struct_img), self.vgg(texture_img)
        loss = 0
        for i in range(len(trans_vgg)):
            if i < 3:
                x_feat_mean = trans_vgg[i].view(trans_vgg[i].size(0), trans_vgg[i].size(1), -1).mean(2)
                y_feat_mean = texture_vgg[i].view(texture_vgg[i].size(0), texture_vgg[i].size(1), -1).mean(2)
                loss += self.criterion(x_feat_mean, y_feat_mean.detach())
            else:
                loss += weights[i] * self.criterion(trans_vgg[i], struct_vgg[i].detach())
        return loss

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
