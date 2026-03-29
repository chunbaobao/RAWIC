import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureGaussian(BaseDistribution):  # TODO
    def __init__(self, ep_params, mix_num=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        N, _, H, W = ep_params.shape
        mean, log_sigma = torch.chunk(ep_params, 2, dim=1)
        self.mean = mean
        self.log_sigma = torch.clamp(log_sigma, min=-7.0)
