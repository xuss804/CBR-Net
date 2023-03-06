from __future__ import division
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv



class BoundaryDetector(nn.Module):
    def __init__(self, in_channel, out_channel=1, stride=1, gamma=1 ,dropout=0.):
        super(BoundaryDetector, self).__init__()
        self.gamma = gamma

        self.contour = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv2d(out_channel, 1, kernel_size=3, padding=1, bias=False)

        )

    def forward(self, x):
        weight = torch.sigmoid(self.contour(x))
        contour = weight*(1-weight)
        x = self.gamma * x * contour + x * weight + x
        return x, weight


class BoundaryRefinementModule(nn.Module):
    def __init__(self, in_dim=64, depth=32, dropout=0.):
        super(BoundaryRefinementModule, self).__init__()

        self.atrous_block3_1 = nn.Sequential(nn.Conv2d(in_dim, depth, 1, 1, padding=0),
                                             nn.BatchNorm2d(depth),
                                             nn.GELU(),
                                             nn.Dropout(dropout),
                                             nn.Conv2d(depth, depth, 1, 1, padding=0),
                                             nn.BatchNorm2d(depth),
                                             nn.GELU(),
                                             nn.Dropout(dropout)
                                             )
        self.atrous_block3_3 = nn.Sequential(nn.Conv2d(in_dim, depth, 3, 1, padding=1),
                                             nn.BatchNorm2d(depth),
                                             nn.GELU(),
                                             nn.Dropout(dropout),
                                             nn.Conv2d(depth, depth, 3, 1, padding=1),
                                             nn.BatchNorm2d(depth),
                                             nn.GELU(),
                                             nn.Dropout(dropout),
                                             )
        self.atrous_block3_5 = nn.Sequential(nn.Conv2d(in_dim, depth, 5, 1, padding=2),
                                             nn.BatchNorm2d(depth),
                                             nn.GELU(),
                                             nn.Dropout(dropout),
                                             nn.Conv2d(depth, depth, 5, 1, padding=2),
                                             nn.BatchNorm2d(depth),
                                             nn.GELU(),
                                             nn.Dropout(dropout),
                                             )

        self.conv_3x3_output = nn.Sequential(nn.Conv2d(depth * 3, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.BatchNorm2d(in_dim),
                                             nn.GELU(),
                                             nn.Dropout(dropout),
                                             nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.BatchNorm2d(in_dim),
                                             nn.GELU(),
                                             nn.Dropout(dropout),
                                             )

        self.BoundaryDetector = BoundaryDetector(in_channel=in_dim, out_channel=1, stride=1, dropout=dropout)

    def forward(self, x):
        atrous_block3_1 = self.atrous_block3_1(x)
        atrous_block3_3 = self.atrous_block3_3(x)
        atrous_block3_5 = self.atrous_block3_5(x)
        out = self.conv_3x3_output(torch.cat([ atrous_block3_1, atrous_block3_3, atrous_block3_5], dim=1))

        out, weight = self.BoundaryDetector(out)

        return out, weight

