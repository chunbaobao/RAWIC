from configs.default import *
from model.gh_as.swin_transformer import PriorCompressor
from model.gh_as.swin_transformer_y_ctx_jpeg_cond import HyperAnalysis, HyperSynthesis, Analysis, Synthesis

from model.base import PriorCompressionModel
from model.base import CondPriorCompressionModel

from model.sp_context.mask_p import MultiMaskedConv2d_P, MaskedConv2d_P_64, CondMaskedConv2d_P_64
from model.sp_context.mask_z import MaskedConv2d


from model.ch_context.group_ch import MultiChannelContext

from model.hyper_fs.fusion import MultiFusion


from model.distribution.lmm import MixtureLogistic as LMM
from model.distribution.rgb_lmm import RGBMixtureLogistic as RGBLMM
from model.distribution.rgb_lmm import MultiRGBMixtureLogistic as MultiRGBLMM
from model.distribution.rggb_lmm import RGGBMixtureLogistic as RGGBLMM
from model.distribution.rggb_lmm_bit import RGGBMixtureLogisticDiffBitdepth as RGGBLMMDB

from model.entropy_models.residual import MultiResidualEntropyModel, ResidualEntropyModel, CondResidualEntropyModel
from model.entropy_model import CustomEntropyBottleneck, GaussianConditional

from model.bit_emb import BitEmb


from model.rawic import RawIC as RawICBE
import utils.misc as misc
import torch.nn as nn

from model.latent_codecs import (
    CondHyperLatentCodec,
    CondHyperpriorLatentCodec,
    CustomHyperLatentCodec,
    CustomHyperpriorLatentCodec,
    CustomGaussianConditionalLatentCodec,
    CustomCheckerboardLatentCodec,
    CustomChannelGroupsLatentCodec,
)

from compressai.latent_codecs import CheckerboardLatentCodec, ChannelGroupsLatentCodec, HyperpriorLatentCodec
from compressai.layers import sequential_channel_ramp, CheckerboardMaskedConv2d, conv1x1

from datasets.dataset import RawJpegLMDBDataset, RawJpegMaskLMDBDataset
from torch.utils.data import ConcatDataset

import torch.optim as optim
from utils.schedulers import WarmupScheduler
import os

batch_size = 64
num_epochs = 600
lr_reduce_patience = 30
lr_reduce_factor = 0.9

# dataset
data_name = [
    # "Canon1DsMkIII",
    # "Canon600D",
    # "NikonD40",
    # "NikonD5200",
    # "OlympusEPL6",
    # "PanasonicGX1",
    "SamsungNX2000",
    # "SonyA57", for weird 15 bit depth
    # "raise",
]
data_paths = [os.path.join("./data", name) for name in data_name]

jpeg_mask_ratio = 1.0

train_dataset = ConcatDataset(
    [
        RawJpegMaskLMDBDataset(data_path, split="train", transform=transform_train, jpeg_mask_ratio=jpeg_mask_ratio)
        for data_path in data_paths
    ]
)
val_dataset = ConcatDataset(
    [
        RawJpegMaskLMDBDataset(data_path, split="val", transform=transform_val, jpeg_mask_ratio=jpeg_mask_ratio)
        for data_path in data_paths
    ]
)


# Model parameters

## Prior compressor
num_ch = 192
prior_ch = 256


g_a = Analysis(4, num_ch)  # rggb
g_s = Synthesis(num_ch, prior_ch)
h_a = HyperAnalysis(num_ch)
h_s = HyperSynthesis(num_ch)


partite = 4
groups = [num_ch // partite for _ in range(partite)]


## y channel context
y_ch_ch_out = 8  # ?


y_ch_ctx = {
    f"y{k}": sequential_channel_ramp(
        sum(groups[:k]),
        groups[k] * y_ch_ch_out,
        min_ch=num_ch,
        num_layers=3,
        make_layer=nn.Conv2d,
        make_act=lambda: nn.LeakyReLU(inplace=True),
        kernel_size=5,
        stride=1,
        padding=2,
    )
    for k in range(1, len(groups))
}


## y spatial context
y_sp_ch_out = 8  # ?

y_sp_ctx = [
    CheckerboardMaskedConv2d(
        groups[k],
        groups[k] * y_sp_ch_out,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    for k in range(len(groups))
]

## feature fusion


fs_ch_out = 2

hyper_fs = [
    sequential_channel_ramp(
        # Input: spatial context, channel context, and hyper params.
        groups[k] * y_sp_ch_out + (k > 0) * groups[k] * y_ch_ch_out + num_ch,
        groups[k] * fs_ch_out,
        min_ch=num_ch * 2,
        num_layers=3,
        make_layer=nn.Conv2d,
        make_act=lambda: nn.LeakyReLU(inplace=True),
        kernel_size=1,
        stride=1,
        padding=0,
    )
    for k in range(len(groups))
]

_latent_codec = {
    f"y{k}": CustomCheckerboardLatentCodec(
        latent_codec={
            "y": CustomGaussianConditionalLatentCodec(),
        },
        context_prediction=y_sp_ctx[k],
        entropy_parameters=hyper_fs[k],
    )
    for k in range(len(groups))
}


latent_codec = CondHyperpriorLatentCodec(
    latent_codec={
        "y": CustomChannelGroupsLatentCodec(
            groups=groups,
            channel_context=y_ch_ctx,
            latent_codec=_latent_codec,
        ),
        "hyper": CondHyperLatentCodec(
            entropy_bottleneck=CustomEntropyBottleneck(num_ch),
            h_a=h_a,
            h_s=h_s,
        ),
    }
)

pri_compressor = CondPriorCompressionModel(
    g_a=g_a,
    g_s=g_s,
    latent_codec=latent_codec,
)


## x spatial context
x_sp_ch_out = 256
# x_sp_ctx = MaskedConv2d_P_64(3, x_sp_ch_out, kernel_size=7, padding=3)
x_sp_ctx = CondMaskedConv2d_P_64(7, x_sp_ch_out, kernel_size=7, padding=3)


## fusion

fs_ch_in = prior_ch + x_sp_ch_out
fs_ch_out = 256

fusion = conv1x1(fs_ch_in, fs_ch_out)


## distribution
no_multichannel_lmm = False
mix_num = 5
distribution = RGGBLMMDB

distribution.no_multichannel_lmm = no_multichannel_lmm
distribution.mix_num = mix_num

## entropy parameter

ep_ch_in = fs_ch_out
ep_ch_out = misc.get_entropy_model_channels(distribution)
ep = CondResidualEntropyModel(ep_ch_in, ep_ch_out)

# bit embedding
start_bit = 9
end_bit = 14
num_bits = end_bit - start_bit + 1
emb_ch = 512
bit_emb = BitEmb(num_bits, emb_ch, 4)

model = RawICBE(
    prior_ic=pri_compressor,
    sp_ctx=x_sp_ctx,
    ep=ep,
    fusion=fusion,
    distribution=distribution,
    bit_emb=bit_emb,
)
model.start_bit = start_bit
model.end_bit = end_bit


if multistep:

    def scheduler_fn(optimizer):
        base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if warmup:
            return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
        return base_scheduler

    scheduler = scheduler_fn
else:

    def scheduler_fn(optimizer):
        base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_reduce_factor,
            patience=lr_reduce_patience,
        )
        if warmup:
            return WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
        return base_scheduler

    scheduler = scheduler_fn
