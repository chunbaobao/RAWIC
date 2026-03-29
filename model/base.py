import torch.nn as nn
from compressai.models import CompressionModel
from compressai.latent_codecs import LatentCodec
import torch
import torch.nn.functional as F


class PriorCompressionModel(CompressionModel):
    """Simple VAE model with arbitrary latent codec.

    .. code-block:: none

               ┌───┐  y  ┌────┐ y_hat ┌───┐
        x ──►──┤g_a├──►──┤ lc ├───►───┤g_s├──►── prior
               └───┘     └────┘       └───┘
    """

    def __init__(self, g_a: nn.Module, g_s: nn.Module, latent_codec: LatentCodec):
        super().__init__()
        self.g_a = g_a
        self.g_s = g_s
        self.latent_codec = latent_codec

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, x):
        y = self.g_a(x)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        prior = self.g_s(y_hat)
        return {
            "prior": prior,
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x):
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)
        return outputs

    def decompress(self, *args, **kwargs):
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out["y_hat"]
        prior = self.g_s(y_hat)
        return {
            "prior": prior,
        }


class CondPriorCompressionModel(CompressionModel):
    def __init__(self, g_a: nn.Module, g_s: nn.Module, latent_codec: LatentCodec):
        super().__init__()
        self.g_a = g_a
        self.g_s = g_s
        self.latent_codec = latent_codec

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, x, cond):
        y = self.g_a(x, cond)
        y_out = self.latent_codec(y, cond)
        y_hat = y_out["y_hat"]
        prior = self.g_s(y_hat, cond)
        return {
            "prior": prior,
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x, cond):
        y = self.g_a(x, cond)
        outputs = self.latent_codec.compress(y, cond)
        return outputs

    def decompress(self, *args, **kwargs):
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out["y_hat"]
        prior = self.g_s(y_hat, kwargs.get("cond", None))
        return {
            "prior": prior,
        }


class JpegConditionedSequential(nn.Sequential):  # from r2lcm
    def __init__(self, *kargs, **kwargs):
        super().__init__(*tuple(filter(lambda x: x is not None, kargs)), **kwargs)

    def forward(self, input, x_jpg, lam_embedding=None):
        for idx, module in enumerate(self):
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                input = module(input)
            else:
                input = module(torch.cat([F.interpolate(x_jpg, size=input.shape[2:]), input], dim=1))
            # input = BoundFunction.apply(input, -10, 10)
        return input
