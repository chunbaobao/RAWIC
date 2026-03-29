import torch
import torch.nn as nn


class BPPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log2 = torch.log(torch.tensor(2.0))

    def forward(self, x, output):
        assert x.ndim == 4  # B, C, H, W
        num_pixels = x.numel() / x.shape[1]
        out = {}
        out["z_bpp"] = -torch.log2(output["likelihoods"]["z"]).sum() / num_pixels
        out["y_bpp"] = -torch.log2(output["likelihoods"]["y"]).sum() / num_pixels
        out["latent_bpp"] = out["z_bpp"] + out["y_bpp"]
        out["x_bpp"] = -output["likelihoods"]["x"].sum() / (self.log2 * num_pixels)
        out["loss"] = out["x_bpp"] + out["latent_bpp"]

        return out


class BPSPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log2 = torch.log(torch.tensor(2.0))

    def forward(self, x, output):
        assert x.ndim == 4  # B, C, H, W
        num_pixels = x.numel()
        out = {}
        out["z_bpp"] = -torch.log2(output["likelihoods"]["z"]).sum() / num_pixels
        out["y_bpp"] = -torch.log2(output["likelihoods"]["y"]).sum() / num_pixels
        out["latent_bpp"] = out["z_bpp"] + out["y_bpp"]
        out["x_bpp"] = -output["likelihoods"]["x"].sum() / (self.log2 * num_pixels)
        out["loss"] = out["x_bpp"] + out["latent_bpp"]

        return out
