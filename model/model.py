import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from .model_util import *


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class SharedEncoder(nn.Module):
    def __init__(self, resnet_name):
        super(SharedEncoder, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        class_num = 241
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        self.feature_layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

        self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
        self.fc.apply(init_weights)
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):

        low = self.layer0(x)
        rec = self.feature_layers(low)
        y = self.avgpool(rec)
        y = y.view(y.size(0), -1)
        y1 = self.fc(y)

        return low, rec, y, y1

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        # b.append(self.bottleneck.parameters())
        b.append(self.fc.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr':  1* learning_rate},
                {'params': self.get_10x_lr_params(),        'lr': 10* learning_rate}]

    def output_num(self):
      return self.__in_features


class PrivateEncoder(nn.Module):
	def __init__(self, input_channels, code_size):
		super(PrivateEncoder, self).__init__()
		self.input_channels = input_channels
		self.code_size = code_size

		self.cnn = nn.Sequential(nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3),
								nn.BatchNorm2d(64),
								nn.ReLU(),
								nn.Conv2d(64, 128, 3, stride=2, padding=1),
								nn.BatchNorm2d(128),
								nn.ReLU(),
								nn.Conv2d(128, 256, 3, stride=2, padding=1),
								nn.BatchNorm2d(256),
								nn.ReLU(),
								nn.Conv2d(256, 256, 3, stride=2, padding=1),
								nn.BatchNorm2d(256),
								nn.ReLU(),
								nn.Conv2d(256, 256, 3, stride=2, padding=1),
								nn.BatchNorm2d(256),
								nn.ReLU())
		self.model = []
		self.model += [self.cnn]
		self.model += [nn.AdaptiveAvgPool2d((1, 1))]
		self.model += [nn.Conv2d(256, code_size, 1, 1, 0)]
		self.model = nn.Sequential(*self.model)


	def forward(self, x):
		bs = x.size(0)
		output = self.model(x).view(bs, -1)

		return output

class PrivateDecoder(nn.Module):
    def __init__(self, shared_code_channel, private_code_size):
        super(PrivateDecoder, self).__init__()
        num_att = 256
        self.shared_code_channel = shared_code_channel
        self.private_code_size = private_code_size

        self.main = []
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            Conv2dBlock(256, 128, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),

            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Conv2dBlock(128, 64 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            Conv2dBlock(64 , 32 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),

            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),       #add    # 56*56
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            Conv2dBlock(32 , 32 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),

            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),        #add   # 112*112
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            Conv2dBlock(32 , 32 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'),

            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh())

        self.main += [Conv2dBlock(shared_code_channel+num_att+1, 256, 3, stride=1, padding=1, norm='ln', activation='relu', pad_type='reflect', bias=False)]
        self.main += [ResBlocks(3, 256, 'ln', 'relu', pad_type='zero')]
        self.main += [self.upsample]

        self.main = nn.Sequential(*self.main)
        self.mlp_att   = nn.Sequential(nn.Linear(private_code_size, private_code_size),
                                nn.ReLU(),
                                nn.Linear(private_code_size, private_code_size),
                                nn.ReLU(),
                                nn.Linear(private_code_size, private_code_size),
                                nn.ReLU(),
                                nn.Linear(private_code_size, num_att))

    def forward(self, shared_code, private_code, d):
        d = Variable(torch.FloatTensor(shared_code.shape[0], 1).fill_(d)).cuda()
        d = d.unsqueeze(1)
        d_img = d.view(d.size(0), d.size(1), 1, 1).expand(d.size(0), d.size(1), shared_code.size(2), shared_code.size(3))
        att_params = self.mlp_att(private_code)
        att_img    = att_params.view(att_params.size(0), att_params.size(1), 1, 1).expand(att_params.size(0), att_params.size(1), shared_code.size(2), shared_code.size(3))
        code         = torch.cat([shared_code, att_img, d_img], 1)

        output = self.main(code)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.feature = nn.Sequential(
            Conv2dBlock(3, 64, 6, stride=2, padding=2, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 1, padding=0),
            # nn.Sigmoid()
        )
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.feature(x)
        return x


class DomainClassifier(nn.Module):
  def __init__(self, in_feature=512, hidden_size=1024):
    super(DomainClassifier, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
