import torch.nn as nn
import torch.nn.functional as F
from unet_utils import Up, Down
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Pathology_block_ATSTNet(nn.Module):
    def __init__(self, in_features, out_features, n_residual_blocks, alt_leak=False, neg_slope=1e-2):
        super(Pathology_block_ATSTNet, self).__init__()

        ext_model = [nn.Conv2d(in_features, out_features, 1),
                       nn.InstanceNorm2d(out_features),
                       nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]
        ext_model += [nn.ReflectionPad2d(1),
                       nn.Conv2d(out_features, out_features, 4, stride=2),
                       nn.InstanceNorm2d(out_features),
                       nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]

        for _ in range(n_residual_blocks):
            ext_model += [ResidualBlock(out_features, alt_leak, neg_slope)]
        self.extractor = nn.Sequential(*ext_model)

    def forward(self, x1, x2, x3):

        x1 = F.interpolate(x1, scale_factor=0.5)
        diffY1 = x2.size()[2] - x1.size()[2]
        diffX1 = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX1 // 2, diffX1 - diffX1 // 2,
                        diffY1 // 2, diffY1 - diffY1 // 2])    # x1: 8, 64, 128, 128  x2: 8, 128, 128, 128
        x = torch.cat([x1, x2], dim=1)    # 8, 192, 128, 128

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        diffY2 = x2.size()[2] - x3.size()[2]
        diffX2 = x2.size()[3] - x3.size()[3]
        x3 = F.pad(x3, [diffX2 // 2, diffX2 - diffX2 // 2,
                        diffY2 // 2, diffY2 - diffY2 // 2])    # x3: 8, 256, 128, 128
        x = torch.cat([x, x3], dim=1)    # 8, 448, 128, 128

        return self.extractor(x)


class Generator_seg_ATSTNet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=10, alt_leak=False, neg_slope=0.1):
        super(Generator_seg_ATSTNet, self).__init__()
        # Initial convolution block [N 32 H W]
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_nc, 32, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Downsampling [N 64 H/2 W/2]
        self.down1 = Down(32, 64, alt_leak, neg_slope)
        # Downsampling [N 128 H/4 W/4]
        self.down2 = Down(64, 128, alt_leak, neg_slope)
        # Downsampling [N 256 H/8 W/8]
        self.down3 = Down(128, 256, alt_leak, neg_slope)


        # Residual blocks [N 256 H/8 W/8]
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.res_blocks_1 = nn.Sequential(*res_ext_encoder)

        # merge features [N 256 H/8 W/8]
        self.pathology_feature = Pathology_block_ATSTNet(448, 256, n_residual_blocks // 2, alt_leak, neg_slope)

        self.merge = nn.Sequential(nn.Conv2d(512, 256, 1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

        # Residual blocks [N 256 H/8 W/8]
        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.res_blocks_2 = nn.Sequential(*res_ext_decoder)


        # Upsampling [N 128 H/4 W/4]
        self.up1 = Up(256, 128, alt_leak, neg_slope)
        # Upsampling [N 64 H/2 W/2]
        self.up2 = Up(128, 64, alt_leak, neg_slope)
        # Upsampling [N 32 H W]
        self.up3 = Up(64, 32, alt_leak, neg_slope)
        # Upsampling [N 3 H W]
        self.out_style_path = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(32, output_nc, 7),
                                  nn.Tanh())

        self.out_pathology_path = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(256, 1, 3),
                                     nn.Sigmoid())


        # Residual blocks [N 256 H/8 W/8]
        res_ext_decoder = []
        for _ in range(6):
            res_ext_decoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.res_blocks_a = nn.Sequential(*res_ext_decoder)
        # Upsampling [N 128 H/4 W/4]
        self.upa1 = Up(256, 128, alt_leak, neg_slope)
        # Upsampling [N 64 H/2 W/2]
        self.upa2 = Up(128, 64, alt_leak, neg_slope)
        # Upsampling [N 32 H W]
        self.upa3 = Up(64, 32, alt_leak, neg_slope)
        # Upsampling [N 3 H W]
        self.out_style_path_a = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(32, 1, 7),
                                  nn.Sigmoid())


    def forward(self, x, mode='O'):    # 8, 3, 512, 512
        # encoder
        x0 = self.inc(x)    # 8, 32, 512, 512
        x1 = self.down1(x0)    # 8, 64, 256, 256
        x2 = self.down2(x1)    # 8, 128, 128, 128
        x3 = self.down3(x2)    # 8, 256, 64, 64

        # extract feature
        pathology_feature = self.pathology_feature(x1, x2, x3)    # 8, 256, 64, 64
        out_pathology_path = self.out_pathology_path(pathology_feature)    # 8, 1, 64, 64

        if mode == 'S':
            latent_feature_a = self.res_blocks_a(x3)    # 8, 256, 64, 64
            features_a = latent_feature_a
            x_a = self.upa1(features_a, x2)    # 8, 128, 128, 128
            x_a = self.upa2(x_a, x1)    # 8, 64, 256, 256
            x_a = self.upa3(x_a, x0)    # 8, 32, 512, 512
            outputs_a = self.out_style_path_a(x_a)    # 8, 1, 512, 512
            return outputs_a

        latent_feature = self.res_blocks_1(x3)    # 2, 256, 64, 64
        features = torch.cat([latent_feature, pathology_feature], dim=1)    # 2, 512, 64, 64
        features = self.merge(features)    # 2, 256, 64, 64
        features = self.res_blocks_2(features)    # 2, 256, 64, 64

        # decoder
        x = self.up1(features, x2)    # 2, 128, 128, 128
        x = self.up2(x, x1)    # 2, 64, 256, 256
        x = self.up3(x, x0)    # 2, 32, 512, 512
        outputs = self.out_style_path(x)    # 2, 3, 512, 512
        return outputs, latent_feature, out_pathology_path, pathology_feature


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1 ),
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0])




