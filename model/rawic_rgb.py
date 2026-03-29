import torch
import torch.nn as nn
import time


class RawIC(nn.Module):
    start_bit = 9
    end_bit = 14

    def __init__(self, prior_ic, sp_ctx, ep, fusion, distribution):
        super().__init__()

        self.prior_ic = prior_ic
        self.sp_ctx = sp_ctx
        self.ep = ep
        self.fusion = fusion
        self.distribution = distribution

    def forward(self, x, rgb):

        prior_out = self.prior_ic(x)
        x = x * 2  # follow dlpr

        sp_ctx = self.sp_ctx(x)
        ctx = self.fusion(torch.cat([prior_out["prior"], sp_ctx], dim=1))
        ep_params = self.ep(ctx)
        x_dist = self.distribution(ep_params)
        x_likelihoods = x_dist(x)

        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": prior_out["likelihoods"]["y"],
                "z": prior_out["likelihoods"]["z"],
            },
        }

    def compress_latent(self, x, rgb):

        latent_code = self.prior_ic.compress(x)
        return latent_code, 0  # dummy bit depth

    def decompress_latent(self, *args, **kwargs):
        return self.prior_ic.decompress(*args, **kwargs)
