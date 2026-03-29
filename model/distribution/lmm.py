import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureLogistic(nn.Module):
    mix_num = 5  # default

    def __init__(self, ep_params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        B, C, H, W = ep_params.shape  # C = num_channels * mix_num * 3
        C = C // self.mix_num // 3
        mean, log_scale, weights = torch.chunk(ep_params, 3, dim=1)
        self.mean = torch.reshape(mean, (B, C, self.mix_num, H, W))
        self.log_scale = torch.clamp(torch.reshape(log_scale, (B, C, self.mix_num, H, W)), min=-7.0)

        self.weights = torch.reshape(weights, (B, C, self.mix_num, H, W))

    def forward(self, input):
        B, C, H, W = input.shape

        half = float(0.5) / 255.0 * 2
        x = input.view(B, C, 1, H, W).expand(-1, -1, self.mix_num, -1, -1)
        centered_x = x - self.mean
        inv_scale = torch.exp(-self.log_scale)

        plus_in = inv_scale * (centered_x + half)
        cdf_plus = torch.sigmoid(plus_in)

        min_in = inv_scale * (centered_x - half)
        cdf_min = torch.sigmoid(min_in)

        log_one_minus_cdf_min = -F.softplus(min_in)  # log(1 - sigmoid(min_in))
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log(sigmoid(plus_in))
        cdf_delta = cdf_plus - cdf_min

        log_probs = torch.where(
            x - half < 0.001,
            log_cdf_plus,
            torch.where(
                x + half > 1.999,
                log_one_minus_cdf_min,
                torch.log(torch.clamp(cdf_delta, min=1e-9)),
            ),
        )

        log_probs = log_probs + self.log_prob_from_logits(self.weights)

        return self.log_sum_exp(log_probs)

    def log_prob_from_logits(self, x):
        axis = 2
        m = torch.amax(x, axis, keepdim=True)
        return x - m - torch.log(torch.sum(torch.exp(x - m), axis, keepdim=True))

    def log_sum_exp(self, x):
        axis = 2
        m = torch.amax(x, axis, keepdim=True)
        return m.squeeze(axis) + torch.log(torch.sum(torch.exp(x - m), axis))
