import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from typing import Type, Callable, Tuple, Optional, Set, List, Union
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath

from BaseBlock import conv3x3, BasicBlock

class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=0.25):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, int(in_planes * ratio), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(int(in_planes * ratio), in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x
        return out

class MBConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride_size=1, expand_rate=4, se_rate=0.25, dropout=0., drop_path=0.):
        super().__init__()

        self.SE = ChannelAttention(in_dim, ratio=se_rate)
        self.conv = BasicBlock(in_dim, in_dim, stride=1, dropout = dropout)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(in_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout)
        # )

    def forward(self, x):
        out = self.SE(x)
        out = self.conv(out) + x
        return out

def window_block(x, window_size=(8, 8)):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//window_size[0], window_size[0], W//window_size[1], window_size[1])
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return x

def window_unblock(x, original_size, window_size=(8, 8)):
    H, W = original_size
    B = int(x.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = x.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output

def grid_block(x, grid_size=(8, 8)):
    B, C, H, W = x.shape
    # Unfold input
    grid = x.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid

def grid_unblock(x, original_size, grid_size=(8, 8)):
    (H, W), C = original_size, x.shape[-1]
    # Compute original batch size
    B = int(x.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = x.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output

class Rel_Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias = True, mlp_ratio=4.,qk_scale = None, drop=0.,attn_drop=0., proj_drop=0.):
        super(Rel_Attention, self).__init__()
        self.dim = dim
        self.win_h, self.win_w = window_size
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = head_dim ** (-0.5)

        self.relative_bias_table = nn.Parameter(torch.zeros((2*self.win_h-1)*(2*self.win_w-1), num_heads))

        coords = torch.meshgrid((torch.arange(self.win_h), torch.arange(self.win_w)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:,:,None]-coords[:, None, :]

        relative_coords[0] += self.win_h - 1
        relative_coords[1] += self.win_w - 1
        relative_coords[0] *= 2 * self.win_w - 1
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.qkv = nn.Conv1d(dim, dim*3, 1, bias=qkv_bias)
        "self.qkv = nn.Linear(in_features=dim, out_features=3 * dim, bias=True)"

        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        # self.proj = nn.Conv2d(dim, dim, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        # self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B_, N, C  = x.shape

        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = relative_bias.reshape(N, N, -1).permute(2, 0, 1).contiguous()

        "qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)"
        qkv = self.qkv(x.transpose(1, 2)).transpose(1, 2)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        """计算余弦相似性"""
        q = torch.divide(q, torch.sqrt(torch.sum(q ** 2, axis=3)).unsqueeze(axis=3))
        k = torch.divide(k, torch.sqrt(torch.sum(k ** 2, axis=3)).unsqueeze(axis=3))
        # q = self.softmax1(q)
        # k = self.softmax1(k)

        """# Scale query
        q = q * self.scale
        # Compute attention maps
        attn = self.softmax(q @ k.transpose(-2, -1) + relative_bias)"""
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn + relative_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        # output = self.proj(output)
        # output = self.proj_drop(output)

        return output

class Window_Block(nn.Module):
    def __init__(self, dim, block_size=(8, 8), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop = 0., drop_path=0., act_layer=nn.GELU, norm_layer=Channel_Layernorm):
        super(Window_Block, self).__init__()
        self.block_size = block_size
        self.norm_1 = nn.LayerNorm(dim)

        self.attn = Rel_Attention(dim, block_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm_2 = nn.LayerNorm(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio * dim), act_layer=act_layer, drop=drop)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout(drop),

            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        assert x.shape[2]%self.block_size[0] == 0 & x.shape[3]%self.block_size[1]==0,'image size should be divisible by block_size'

        B, C, H, W = x.shape
        out = window_block(x, self.block_size).view(-1, self.block_size[0] * self.block_size[1], C)
        out = out + self.drop_path(self.attn(self.norm_1(out)))

        # output = out + self.drop_path(self.mlp(self.norm_2(out)))
        out = window_unblock(self.norm_2(out), (H, W), self.block_size)
        out = self.mlp(out) + out
        return out

class Grid_Block(nn.Module):
    def __init__(self, dim, grid_size=(8, 8), num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=Channel_Layernorm):
        super(Grid_Block, self).__init__()
        self.grid_size = grid_size
        self.norm_1 = nn.LayerNorm(dim)
        self.attn = Rel_Attention(dim, grid_size, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = nn.LayerNorm(dim)
        """self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio * dim), act_layer=act_layer, drop=drop)"""
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout(drop),

            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        assert x.shape[2]%self.grid_size[0]==0 & x.shape[3]%self.grid_size[1]==0,  'image size should be divisible by grid_size'

        B, C, H, W = x.shape
        out = grid_block(x, self.grid_size).view(-1, self.grid_size[0] * self.grid_size[1], C)
        out = out + self.drop_path(self.attn(self.norm_1(out)))

        """output = out + self.drop_path(self.mlp(self.norm_2(out)))
        """
        out = grid_unblock(self.norm_2(out), (H, W), self.grid_size)
        out = self.mlp(out) + out
        return out

class Max_Block(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, block_size=(8, 8), grid_size=(8, 8),
                 mbconv_ksize=3, pooling_size=1, mbconv_expand_rate=4, se_reduce_rate=0.25,
                 mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=Channel_Layernorm):
        super().__init__()
        self.mbconv = MBConv(in_dim, out_dim, mbconv_ksize, pooling_size, mbconv_expand_rate, se_reduce_rate, drop)
        self.block_attn = Window_Block(out_dim, block_size, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                                       attn_drop, drop_path, act_layer, norm_layer)
        self.grid_attn = Grid_Block(out_dim, grid_size, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                                    attn_drop, drop_path, act_layer, norm_layer)

    def forward(self, x):
        x = self.mbconv(x)
        x = self.block_attn(x)
        x = self.grid_attn(x)
        return x


