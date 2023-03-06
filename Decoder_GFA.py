import torch
import torch.nn as nn
import torch.nn.functional as F

from GFA import MBConv, Window_Block, Grid_Block, Max_Block, Channel_Layernorm
from BaseBlock import BoundaryWiseAttentionGate2D, BasicBlock

class upsample(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1, num_heads=8, block_size=(8, 8), grid_size=(8, 8),
                 mbconv_ksize=3, pooling_size=1, mbconv_expand_rate=2, se_reduce_rate=0.25,
                 mlp_ratio=2, dropout=0., attn_drop=0., drop_path=0.):
        super(upsample, self).__init__()
        # in_size表示输入通道数， out_size表示输出通道数

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_dim)

        layers = []
        for _ in range(0, num_layers):
            layers.append(Max_Block(in_dim=out_dim, out_dim=out_dim, num_heads=num_heads, block_size=block_size, grid_size=grid_size,
                 mbconv_ksize=mbconv_ksize, pooling_size=pooling_size, mbconv_expand_rate=mbconv_expand_rate, se_reduce_rate=se_reduce_rate,
                 mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None, drop=dropout, attn_drop=attn_drop, drop_path=drop_path,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm))

        self.layer = nn.Sequential(*layers)

        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.contour = BoundaryWiseAttentionGate2D(out_dim)

    def forward(self, inputs1, inputs2):

        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu(outputs)

        outputs = self.layer(outputs)

        return outputs
