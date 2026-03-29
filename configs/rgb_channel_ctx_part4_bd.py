from configs.default import *
from model.gh_as.swin_transformer_y_ctx import HyperAnalysis, HyperSynthesis, Analysis, Synthesis

from model.base import PriorCompressionModel

from model.sp_context.mask_p import MaskedConv2d_P_64


from model.distribution.lmm import MixtureLogistic as LMM
from model.distribution.rgb_lmm import RGBMixtureLogistic as RGBLMM
from model.distribution.rgb_lmm import MultiRGBMixtureLogistic as MultiRGBLMM
from model.distribution.rgb_lmm_bit_pw import RGBMixtureLogisticDiffBitdepthPixelWise as RGBLMMDBPW
from model.distribution.rgb_lmm_bit import RGBMixtureLogisticDiffBitdepth as RGBLMMDB
from model.distribution.rggb_lmm import RGGBMixtureLogistic as RGGBLMM
from model.distribution.rggb_lmm_bit import RGGBMixtureLogisticDiffBitdepth as RGGBLMMDB
from model.distribution.rggb_lmm_bit_pw import RGGBMixtureLogisticDiffBitdepthPixelWise as RGGBLMMDBPW


from model.entropy_models.residual import ResidualEntropyModel
from model.entropy_model import CustomEntropyBottleneck

from model.rawic_rgb import RawIC as RawICRGB
from model.rawic_wo_jpeg import RawIC as RawICWJ

import utils.misc as misc

from model.bit_emb import BitEmb
from model.bit_emb import BitEmbPW

from utils.sampler import ConstantSampler

import torch.nn as nn

from model.latent_codecs import (
    CustomHyperLatentCodec,
    CustomGaussianConditionalLatentCodec,
    CustomCheckerboardLatentCodec,
    CustomChannelGroupsLatentCodec,
)

from compressai.latent_codecs import HyperpriorLatentCodec
from compressai.layers import sequential_channel_ramp, CheckerboardMaskedConv2d, conv1x1

from datasets.dataset import ImgDataset

from datasets.transform import TrainTransform, EvalTransform
import torch.optim as optim
from utils.schedulers import WarmupScheduler

# dataset

train_path = "/NEW_EDS/JJ_Group/zhengch2506/nasic/data/DIV2K_train_p128"
val_path = "/NEW_EDS/JJ_Group/zhengch2506/nasic/data/DIV2K_valid_p128"
p_hflip = 0.5
p_vflip = 0.5


transform_train = TrainTransform(patch_sz, p_hflip, p_vflip)
transform_val = EvalTransform(patch_sz)
try:
    train_dataset = ImgDataset(train_path, transform=transform_train)
    val_dataset = ImgDataset(val_path, transform=transform_val)
except Exception as e:
    print(f"Error loading datasets: {e}")
batch_size = 64
num_epochs = 1000
lr_reduce_patience = 30
lr_reduce_factor = 0.9

# Model parameters

## Prior compressor
num_ch = 192
prior_ch = 256


g_a = Analysis(3, num_ch)
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


latent_codec = HyperpriorLatentCodec(
    latent_codec={
        "y": CustomChannelGroupsLatentCodec(
            groups=groups,
            channel_context=y_ch_ctx,
            latent_codec=_latent_codec,
        ),
        "hyper": CustomHyperLatentCodec(
            entropy_bottleneck=CustomEntropyBottleneck(num_ch),
            h_a=h_a,
            h_s=h_s,
        ),
    }
)

pri_compressor = PriorCompressionModel(
    g_a=g_a,
    g_s=g_s,
    latent_codec=latent_codec,
)


## x spatial context
x_sp_ch_out = 256
x_sp_ctx = MaskedConv2d_P_64(3, x_sp_ch_out, kernel_size=7, padding=3)


## fusion

fs_ch_in = prior_ch + x_sp_ch_out
fs_ch_out = 256

hyper_fs = conv1x1(fs_ch_in, fs_ch_out)


## distribution
no_multichannel_lmm = False
mix_num = 5
distribution = RGBLMMDB

distribution.mix_num = mix_num

## entropy parameter

ep_ch_in = fs_ch_out
ep_ch_out = misc.get_entropy_model_channels(distribution)
ep = ResidualEntropyModel(ep_ch_in, ep_ch_out)

# bit embedding
start_bit = 5
end_bit = 8
num_bits = end_bit - start_bit + 1
emb_ch = 512
bit_emb = BitEmb(num_bits, emb_ch, 3)


model = RawICWJ(
    prior_ic=pri_compressor,
    sp_ctx=x_sp_ctx,
    ep=ep,
    fusion=hyper_fs,
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
