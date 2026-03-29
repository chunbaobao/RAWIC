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
        x, bit_depth = self.norm_by_bit_depth(x)
        bit_depth = torch.ones_like(bit_depth) * self.end_bit  #

        prior_out = self.prior_ic(x, rgb)
        x = x * 2
        sp_ctx = self.sp_ctx(x, rgb)
        ctx = self.fusion(torch.cat([prior_out["prior"], sp_ctx], dim=1))

        ep_params = self.ep(ctx, rgb)
        x_dist = self.distribution(ep_params)
        x_likelihoods = x_dist(x, bit_depth)

        return {
            "likelihoods": {
                "x": x_likelihoods,
                "y": prior_out["likelihoods"]["y"],
                "z": prior_out["likelihoods"]["z"],
            },
        }

    def norm_by_bit_depth(self, x: torch.Tensor):
        max_vals = x.flatten(start_dim=1).max(dim=1).values
        bit_depth = torch.ceil(torch.log2(max_vals + 1))
        bit_depth = bit_depth.clip(self.start_bit, self.end_bit)
        x_norm = x / (2 ** bit_depth.reshape(-1, 1, 1, 1) - 1)
        return x_norm, bit_depth

    def compress_latent(self, x, rgb):
        x, bit_depth = self.norm_by_bit_depth(x)
        bit_depth = torch.ones_like(bit_depth) * self.end_bit  #
        latent_code = self.prior_ic.compress(x, rgb)
        return latent_code, bit_depth

    def decompress_latent(self, *args, **kwargs):
        return self.prior_ic.decompress(*args, **kwargs)

    # def compress(self, x: torch.Tensor, rgb: torch.Tensor):
    #     """
    #     input x: (B, 4, H, W)  after patchify
    #     input rgb: (B, 3, H, W)  jpg patch

    #     """
    #     x_stream = []
    #     results = {}
    #     B = x.shape[0]
    #     COT = self.sp
    def get_bit_depth_num(self):
        return 1
