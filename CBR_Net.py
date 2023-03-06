import torch
import torch.nn as nn
import torch.nn.functional as F

from BaseBlock import downsample, BasicBlock
from Decoder_GFA import upsample
from UBR import BoundaryRefinementModule

class CBR_Net(nn.Module):
    def __init__(self, num_classes=1, num_layers=[1, 1, 0, 0], block_size=[8, 8, 8, 8], grid_size=[8, 8, 8, 8], heads = [16, 8, 4, 4],se_reduce_rate = [0.5, 0.5, 0.75, 0.75], transformers_type_index=0,
                 hidden_features=128, attn_drop=0., proj_drop=0.,dropout=0., reduce_size=8, projection = "interp", rel_pos="True"):
        super(CBR_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.down1 = BasicBlock(32, 64, stride=2, dropout = dropout)
        self.down2 = BasicBlock(64, 128, stride=2, dropout = dropout)
        self.down3 = BasicBlock(128, 128, stride=2, dropout = dropout)
        self.down4 = BasicBlock(128, 256, stride=2, dropout = dropout)

        self.up1 = upsample(in_dim=384, out_dim=128, num_layers=num_layers[0], num_heads=heads[0], block_size=(block_size[0], block_size[0]), grid_size=(grid_size[0], grid_size[0]),
                            mbconv_ksize=3, pooling_size=1, mbconv_expand_rate=2, se_reduce_rate=se_reduce_rate[0], mlp_ratio=2, dropout=dropout, attn_drop=0., drop_path=0.)
        self.up2 = upsample(in_dim=256, out_dim=128, num_layers=num_layers[1], num_heads=heads[1], block_size=(block_size[1], block_size[1]), grid_size=(grid_size[1], grid_size[1]),
                            mbconv_ksize=3, pooling_size=1, mbconv_expand_rate=2, se_reduce_rate=se_reduce_rate[1], mlp_ratio=2, dropout=dropout, attn_drop=0., drop_path=0.)
        self.up3 = upsample(in_dim=192, out_dim=64, num_layers=num_layers[2], num_heads=heads[2], block_size=(block_size[2], block_size[2]), grid_size=(grid_size[2], grid_size[2]),
                            mbconv_ksize=3, pooling_size=1, mbconv_expand_rate=2, se_reduce_rate=se_reduce_rate[2], mlp_ratio=2, dropout=dropout, attn_drop=0., drop_path=0.)
        self.up4 = upsample(in_dim=96, out_dim=32, num_layers=num_layers[3], num_heads=heads[3], block_size=(block_size[3], block_size[3]), grid_size=(grid_size[3], grid_size[3]),
                            mbconv_ksize=3, pooling_size=1, mbconv_expand_rate=2, se_reduce_rate=se_reduce_rate[3], mlp_ratio=2, dropout=dropout, attn_drop=0., drop_path=0.)

        self.Boundary128 = BoundaryRefinementModule(in_dim=64, depth=32, dropout=dropout)
        self.Boundary256 = BoundaryRefinementModule(in_dim=32, depth=16, dropout=dropout)
        self.conv = BasicBlock(32, num_classes)  # 32, 1  (64, 1)

        self._initialize_weights()

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        x0 = self.dropout(self.relu(self.bn1(self.conv1(x))))  # feature1 torch.Size([8, c0, 128, 128])
        x0 = self.dropout(self.relu(self.bn3(self.conv3(x0))))

        feature1 = self.down1(x0)  # feature1 torch.Size([8, c1, 128, 128])
        feature2 = self.down2(feature1)  # feature2 torch.Size([8, c2, 64, 64])
        feature3 = self.down3(feature2)  # feature3 torch.Size([8, c3, 32, 32])
        feature4 = self.down4(feature3)  # feature4 torch.Size([8, c4, 16, 16])


        x1 = self.up1(feature3, feature4)  # torch.Size([8, 128, 32, 32])  contour1 torch.Size([8, 1, 32, 32])
        x2 = self.up2(feature2, x1)  # torch.Size([8, 128, 64, 64])  contour2 torch.Size([8, 1, 64, 64])
        x3 = self.up3(feature1, x2)  # torch.Size([8, 128, 128, 128])  contour3 torch.Size([8, 1, 128, 128])

        mask_feature128, mask128 = self.Boundary128(x3)
        x4 = self.up4(x0, mask_feature128)
        mask_feature256, mask256 = self.Boundary256(x4)
        x5 = F.interpolate(mask_feature256, size=x.shape[-2:], mode='bilinear', align_corners=True)

        final_y = self.conv(x5)

        return  final_y, mask128, mask256


    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

from thop import profile
if __name__ == "__main__":
    model = CBR_Net(num_classes = 1).cuda()
    d = torch.rand(8, 3, 512, 512).cuda()
    final_y, mask128, mask256 = model(d)

    print("final_y", final_y.shape)
    print("mask128", mask128.shape)
    print("mask256", mask256.shape)

    macs, params = profile(model, inputs=(d,))
    print('macs:', macs / 1000000000)
    print('params:', params / 1000000)