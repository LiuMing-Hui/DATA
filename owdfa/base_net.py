import warnings
import numpy as np
import timm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from owdfa.utils import orthogonal_tensor


__all__ = ['BinaryClassifier']

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)

warnings.filterwarnings("ignore")


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)

        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size

        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))

        x = avg(x)
        return x


class BaseClassifier_old(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes=20,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='no',
                 pretrained=False,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck

        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.pool = AdaptiveAvgPool2dCustom((1, 1))

        self.dropout = nn.Dropout(drop_rate)

        if self.neck == 'bnneck':
            logger.info('Using BNNeck')
            self.bottleneck = nn.BatchNorm1d(self.num_features)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.fc2 = nn.Linear(
                self.num_features, self.num_classes, bias=False)
            self.bottleneck.apply(weights_init_kaiming)
            self.fc2.apply(weights_init_classifier)
        else:
            self.fc2 = nn.Linear(self.num_features, self.num_classes)

    def forward_featuremaps(self, x):
        featuremap = self.encoder.forward_features(x)
        return featuremap

    def forward_features(self, x):
        featuremap = self.encoder.forward_features(x)
        feature = self.pool(featuremap).flatten(1)

        if self.neck == 'bnneck':
            feature = self.bottleneck(feature)

        return feature

    def forward(self, x, label=None):
        feature = self.forward_features(x)

        x = self.dropout(feature)
        method = self.fc2(x)

        y = method

        if self.is_feat:
            return y, feature

        return y


class BaseClassifier(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes=20,
                 num_patch=3,
                 drop_rate=0.2,
                 is_feat=False,
                 neck='bnneck',
                 pretrained=False,
                 disentangle=False,
                 **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_patch = num_patch
        self.drop_rate = drop_rate
        self.is_feat = is_feat
        self.neck = neck
        self.disentangle = disentangle

        self.encoder = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

        if hasattr(self.encoder, 'get_classifier'):
            self.num_features = self.encoder.get_classifier().in_features
        else:
            self.num_features = self.encoder.last_channel

        self.pool = AdaptiveAvgPool2dCustom((1, 1))
        self.part = AdaptiveAvgPool2dCustom((self.num_patch, self.num_patch))

        self.c_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, stride=64, padding=0),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(drop_rate)

        self.bn_sp = nn.BatchNorm1d(self.num_features)
        self.bn_ge = nn.BatchNorm1d(self.num_features)
        self.bn_sp.bias.requires_grad_(False)  # no shift
        self.bn_ge.bias.requires_grad_(False)

        self.fc2 = nn.Linear(self.num_features, self.num_classes, bias=False)
        self.fc1 = nn.Linear(self.num_features, 2, bias=False)

        self.bn_sp.apply(weights_init_kaiming)
        self.bn_ge.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)
        self.fc1.apply(weights_init_classifier)

        for i in range(self.num_patch ** 2):
            name = 'bnneck' + str(i)
            setattr(self, name, nn.BatchNorm1d(self.num_features))
            getattr(self, name).bias.requires_grad_(False)
            getattr(self, name).apply(weights_init_kaiming)

        sp_dim = self.num_features
        fc = [EqualLinear(sp_dim, sp_dim)]
        for i in range(2):
            fc.append(EqualLinear(sp_dim, sp_dim))
        self.mlp_sp = nn.Sequential(*fc)

        ge_dim = self.num_features
        fc = [EqualLinear(sp_dim, sp_dim)]
        for i in range(2):
            fc.append(EqualLinear(sp_dim, sp_dim))
        fc.append(EqualLinear(sp_dim, ge_dim))          # 变更维度
        self.mlp_ge = nn.Sequential(*fc)

    def forward_featuremaps(self, x):
        featuremap = self.encoder.forward_features(x)   # resnet50
        return featuremap

    def forward_features(self, x, types):
        featuremap = self.encoder.forward_features(x)
        f_g = self.pool(featuremap).flatten(1)

        # disentangle
        f_g_sp = self.mlp_sp(f_g)  # 2048
        f_g_ge = self.mlp_ge(f_g)  # 2048

        # disentangle
        orth_fake = None
        if self.training and self.disentangle:
            b = f_g_ge.size(0)
            f_g_or = f_g_ge.detach().clone().unsqueeze(2).transpose(1, 2)  # [b,1,2048]

            f_g_or_fake = f_g_or[:b//2][types == 0]
            f_g_or_fake = self.c_conv(f_g_or_fake)                         # [b/2[fake],8,128]
            ones = torch.eye(32).cuda()

            orth_fake = torch.mean(f_g_or_fake, dim=0).detach()
            orth_fake = torch.cat((ones, orth_fake), 1)
            orth_fake = orthogonal_tensor(orth_fake).reshape(1, -1)        # [1,2048]

        f_g_sp = self.bn_sp(f_g_sp)
        f_g_ge = self.bn_ge(f_g_ge)

        f_p  = self.part(featuremap)
        f_p  = f_p.view(f_p.size(0), f_p.size(1), -1)                      # [b,c,3*3]
        fs_p = []
        for i in range(self.num_patch ** 2):
            f_p_i = f_p[:, :, i]
            f_p_i = getattr(self, 'bnneck' + str(i))(f_p_i)
            fs_p.append(f_p_i)
        fs_p = torch.stack(fs_p, dim=-1)

        return (f_g_sp, fs_p, f_g_ge), orth_fake

    def forward(self, x, types=None):
        b = x.size(0)
        feature, orth_fake = self.forward_features(x, types)
        (f_g_sp, _, f_g_ge) = feature

        # f_g = self.dropout(f_g)
        method = self.fc2(f_g_sp)

        if self.training and self.disentangle:
            face_type = self.fc1(f_g_ge)
            type_2 = face_type[b//2:].max(1)[1]
        else:
            face_type = None
            type_2 = None

        if self.is_feat:
            return method, face_type, feature, orth_fake, type_2

        return method
