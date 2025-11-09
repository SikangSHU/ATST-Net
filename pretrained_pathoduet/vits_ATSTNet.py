# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
from collections import OrderedDict

from pretrained_pathoduet.timm.models.vision_transformer import Conv2DBlock, Deconv2DBlock, VisionTransformer, _cfg
from pretrained_pathoduet.timm.models.layers.helpers import to_2tuple
from pretrained_pathoduet.timm.models.layers import PatchEmbed

__all__ = [
    'vit_small', 
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
]


class ResidualBlock(nn.Module):
    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features),
                    nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class VisionTransformerMoCo_ATSTNet(VisionTransformer):
    def __init__(self, pretext_token=True, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # inserting a new token
        self.num_prefix_tokens += (1 if pretext_token else 0)
        # self.num_prefix_tokens: 2
        self.pretext_token = nn.Parameter(torch.ones(1, 1, self.embed_dim)) if pretext_token else None
        # self.pretext_token: 1, 1, 768
        embed_len = self.patch_embed.num_patches if self.no_embed_class else self.patch_embed.num_patches + 1
        # embed_len: 197
        embed_len += 1 if pretext_token else 0
        # embed_len: 198
        self.embed_len = embed_len

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pretext_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

        self.drop_rate = 0
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        # version with shared skip_connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # skip connection after positional encoding, shape should be H, W, 64
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # skip connection 1
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # skip connection 2
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # skip connection 3

        self.generated_IHC_decoder = self.create_upsampling_branch(3)  # TODO

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token', 'pretext_token'}

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((self.pretext_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def _ref_embed(self, ref):
        B, C, H, W = ref.shape
        ref = self.patch_embed.proj(ref)
        if self.patch_embed.flatten:
            ref = ref.flatten(2).transpose(1, 2)  # BCHW -> BNC
        ref = self.patch_embed.norm(ref)
        return ref

    def _pos_embed_with_ref(self, x, ref):
        pretext_tokens = self.pretext_token.expand(x.shape[0], -1, -1) * 0 + ref
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.pretext_token is not None:
                x = torch.cat((pretext_tokens, x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x, ref=None):  # x: 2, 3, 224, 224
        x = self.patch_embed(x)  # x: 2, 196, 768
        if ref is None:
            x = self._pos_embed(x)  # x: 2, 198, 768
        else:
            ref = self._ref_embed(ref).mean(dim=1, keepdim=True)
            x = self._pos_embed_with_ref(x, ref)

        extracted_layers = []
        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            if depth + 1 in [3, 6, 9, 12]:
                extracted_layers.append(x)

        x = self.norm(x)  # x: 2, 198, 768
        return x, extracted_layers

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 1]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, ref=None):
        # x: 2, 3, 512, 512
        x_forward_features, z = self.forward_features(x, ref)
        # x_forward_features: 2, 198, 768   Z: 4 [2, 198, 768] for each
        z0, z1, z2, z3, z4 = x, *z
        # z0: 2, 3, 224, 224   z1: 2, 198, 768
        # performing reshape for the convolutional layers and upsampling (restore spatial dimension)
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z1 = z1[:, 2:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)  # z1: 2, 768, 14, 14
        z2 = z2[:, 2:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)  # z2: 2, 768, 14, 14
        z3 = z3[:, 2:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)  # z3: 2, 768, 14, 14
        z4 = z4[:, 2:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim)  # z4: 2, 768, 14, 14

        b4 = self.generated_IHC_decoder.bottleneck_upsampler(z4)  # b4: 2, 512, 28, 28
        b3 = self.decoder3(z3)  # b3: 2, 512, 28, 28
        b3 = self.generated_IHC_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))  # b3: 2, 256, 56, 56
        b2 = self.decoder2(z2)  # b2: 2, 256, 56, 56
        b2 = self.generated_IHC_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))  # b2: 2, 128, 112, 112
        b1 = self.decoder1(z1)  # b1: 2, 128, 112, 112
        b1 = self.generated_IHC_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))  # b1: 2, 64, 224, 224
        output = self.generated_IHC_decoder.decoder0_header(b1)  # 2, 3, 224, 224

        return output

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create Upsampling branch

        Args:
            num_classes (int): Number of output classes

        Returns:
            nn.Module: Upsampling path
        """
        res_ext_encoder = []
        for _ in range(5):
            res_ext_encoder += [ResidualBlock(512, alt_leak=False, neg_slope=0.1)]
        bottleneck_upsampler = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.embed_dim,
                out_channels=self.bottleneck_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            *res_ext_encoder,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(
                self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            Conv2DBlock(
                self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate
            ),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh()
        )

        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        # h: 14, w: 14
        grid_w = torch.arange(w, dtype=torch.float32)
        # grid_w: tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.])
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        # grid_w: 14, 14   grid_h: 14, 14
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        # self.embed_dim: 768
        pos_dim = self.embed_dim // 4  # pos_dim: 192
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim  # omega: 192,
        omega = 1. / (temperature ** omega)  # omega: 192,
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])  # out_w: 196, 192
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])  # out_h: 196, 192
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
        # 1, 196, 768
        assert self.num_prefix_tokens == 2, 'Assuming two and only two tokens, [pretext][cls]'
        pe_token = torch.zeros([1, 2, self.embed_dim], dtype=torch.float32)  # pe_token: 1, 2, 768
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))  # self.pos_embed: 1, 198, 768
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    model = VisionTransformerMoCo_ATSTNet(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo_ATSTNet(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo_ATSTNet(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo_ATSTNet(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model