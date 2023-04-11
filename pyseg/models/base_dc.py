import torch
from torch.functional import unique
import torch.nn as nn
from torch.nn import functional as F

def get_syncbn():
    #return nn.BatchNorm2d
    return nn.SyncBatchNorm

class MEP(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, in_planes, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), proj_dim = 128):
        super(MEP, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        # 尺度一
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[0], dilation=dilations[0], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.proj3 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_planes, proj_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(proj_dim),
                                   nn.ReLU(inplace=True))
        # 尺度二
        self.conv4 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[1], dilation=dilations[1], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.proj4 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_planes, proj_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(proj_dim),
                                   nn.ReLU(inplace=True))
        # 尺度三
        self.conv5 = nn.Sequential(nn.Conv2d(in_planes, inner_planes, kernel_size=3,
                                   padding=dilations[2], dilation=dilations[2], bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True))
        self.proj5 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(inner_planes),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inner_planes, proj_dim, kernel_size=1, padding=0, dilation=1, bias=False),
                                   norm_layer(proj_dim),
                                   nn.ReLU(inplace=True))

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        proj3 = self.proj3(feat3)
        feat4 = self.conv4(x)
        proj4 = self.proj4(feat4)
        feat5 = self.conv5(x)
        proj5 = self.proj5(feat5)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out, proj3, proj4, proj5