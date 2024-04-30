import torch
import random
import math
from torch import nn
from basicsr.archs.stylegan2_arch import (ConvLayer, EqualConv2d, EqualLinear, ResBlock, ScaledLeakyReLU)
from basicsr.utils.registry import ARCH_REGISTRY

from .gfpganv1_arch import GFPGANv1, StyleGAN2GeneratorSFT, ResUpBlock

@ARCH_REGISTRY.register()
class GFPGANv1Guided(nn.Module):
    def __init__(
            self,
            out_size,
            num_style_feat=512,
            parse_n_classes=None,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):
        super(GFPGANv1Guided, self).__init__()
        '''
            Stolen from GFPGANv1. Param parse_n_classes added.
        '''
        
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = ConvLayer(3, channels[f'{first_out_size}'], 1, bias=True, activate=True)

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels

        self.final_conv = ConvLayer(in_channels, channels['4'], 3, bias=True, activate=True)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResUpBlock(in_channels, out_channels))
            in_channels = out_channels

        # to RGB and to parse map
        self.toRGB = nn.ModuleList()
        self.toParseMap = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            # RGB image reconstruction space
            self.toRGB.append(EqualConv2d(channels[f'{2**i}'], 3, 1, stride=1, padding=0, bias=True, bias_init_val=0))
            # Parse map prediction space. The background class is included in every feature map.
            if parse_n_classes is not None:
                self.toParseMap.append(EqualConv2d(channels[f'{2**i}'], parse_n_classes, 1, stride=1, padding=0, bias=True, bias_init_val=0))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        self.final_linear = EqualLinear(
            channels['4'] * 4 * 4, linear_out_channel, bias=True, bias_init_val=0, lr_mul=1, activation=None)

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
            narrow=narrow,
            sft_half=sft_half)

        # load pre-trained stylegan2 model if necessary
        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(decoder_load_path, map_location=lambda storage, loc: storage)['params_ema'])
        # fix decoder without updating params
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=1)))
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0)))
            
    def forward(self, x, return_latents=False, return_rgb=True, return_parse=True, randomize_noise=True, **kwargs):
        """Forward function for GFPGANv1Guided. Same as forward of GFPGANv1.

        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        unet_skips = []
        out_rgbs = []
        out_parse_maps = []

        # encoder
        # 1x1 conv
        feat = self.conv_body_first(x)
        # print('encoder, conv bodyf first: ', feat.shape)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
            # print('conv body down, i to log_size-2, i=', i, feat.shape)

        # 1x1 conv(narrow=1)
        feat = self.final_conv(feat)
        # print('', feat.shape)

        # style code
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        # print('style code', style_code.shape)
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)
            # print('different_w=true, style_code', style_code.shape)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            feat = feat + unet_skips[i]
            # ResUpLayer
            feat = self.conv_body_up[i](feat)
            # print('decode, conv body up, i to log_size-2, i=', i, feat.shape)
            # generate scale and shift for SFT layers
            scale = self.condition_scale[i](feat)
            # print('scale', scale.shape)
            conditions.append(scale.clone())

            shift = self.condition_shift[i](feat)
            # print('shift', shift.shape)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))
            if return_parse:
                out_parse_maps.append(self.toParseMap[i](feat))

        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)
        # print('out', image.shape)
        return image, out_rgbs, out_parse_maps