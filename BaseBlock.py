import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BoundaryWiseAttentionGate2D(nn.Sequential):  #BoundaryWiseAttentionGate2D
    def __init__(self, in_channels, hidden_channels = None):
        super(BoundaryWiseAttentionGate2D,self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 1, kernel_size=1))
    def forward(self, x):
        " x.shape: B, C, H, W "
        " return: feature, weight (B,C,H,W) "
        weight = torch.sigmoid(super(BoundaryWiseAttentionGate2D,self).forward(x))# weight shape: [2, 1, 32, 32]
        x = x * weight + x
        return x, weight


class BoundaryAwareModule(nn.Module):  #BoundaryWiseAttentionGate2D
    def __init__(self, dropout = 0.):
        super(BoundaryAwareModule,self).__init__()

        self.down2 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size=1, padding=1, bias=False)
        self.down3 = nn.Conv2d(128, 32, kernel_size=1, padding=1, bias=False)
        self.down4 = nn.Conv2d(128, 32, kernel_size=1, padding=1, bias=False)

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(160),
            conv3x3(160, 64, stride = 1),
            nn.BatchNorm2d(64),  # inplanes
            nn.GELU(),
            nn.Dropout(dropout),

            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            conv3x3(64, 32, stride=1),
            nn.BatchNorm2d(32),  # inplanes
            nn.GELU(),
            nn.Dropout(dropout),

            conv3x3(32, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.conv = conv3x3(1, 1, stride=1)

    def forward(self, x1, x2, x3, x4, y):
        " x.shape: B, C, H, W "

        x2 = self.down2(x2)
        x3 = self.down3(x3)
        x4 = self.down4(x4)

        x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x1.shape[-2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4], 1)
        x = self.conv1(x) + x1

        x = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=True)
        weight = self.conv2(x)

        predict = self.conv(y * weight + y)

        return predict, weight

class downsample(nn.Module):
    def __init__(self, in_size, out_size, dropout = 0.):
        super(downsample, self).__init__()
        # in_size表示输入通道数， out_size表示输出通道数
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        return outputs

class upsample(nn.Module):
    def __init__(self, in_size, out_size, dropout = 0.):
        super(upsample, self).__init__()
        # in_size表示输入通道数， out_size表示输出通道数

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_size)
        # ！！2D双线性插值法上采样，但是没有改变通道数，与原始论文不一样！！
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.contour = BoundaryWiseAttentionGate2D(out_size)

    def forward(self, inputs1, inputs2):
        # h1,w1 = inputs1.shape[2:]
        # h2, w2 = inputs2.shape[2:]
        # if h1 != h2 or w1 != w2:
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # else:
        #     outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)

        # outputs, contours = self.contour(outputs)

        # outputs = self.dropout(outputs)
        return outputs
        # return outputs,contours

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dropout=0.):
        super().__init__()
        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.bn1 = nn.BatchNorm2d(inplanes) # inplanes
        self.relu = nn.GELU()

        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x):

        residue = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residue)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    # make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=3, stride=1, dilation=2)
    # blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.gelu(self.bn1( self.conv1(x)))  # (shape: (batch_size, channels, h, w))
        out = F.gelu(self.bn2(self.conv2(out)))  # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(out))  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x)  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.gelu(out )  # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out



if __name__ == "__main__":
    # model = upsample(256+128, 128).cuda()
    # d1 = torch.rand(8, 256, 32, 32).cuda()
    # d2 = torch.rand(8, 128, 64, 64).cuda()
    # x1 = model(d2, d1)
    # print("x1", x1.shape)

    model = BoundaryAwareModule().cuda()
    x1 = torch.rand(8, 64, 128, 128).cuda()
    x2 = torch.rand(8, 64, 64, 64).cuda()
    x3 = torch.rand(8, 128, 32, 32).cuda()
    x4 = torch.rand(8, 128, 16, 16).cuda()
    y = torch.rand(8, 1, 512, 512).cuda()
    predict, weight = model(x1, x2, x3, x4, y)
    print("predict", predict.shape)
    print("weight", weight.shape)
