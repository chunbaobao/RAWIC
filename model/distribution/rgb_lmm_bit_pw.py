import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBMixtureLogisticDiffBitdepthPixelWise(nn.Module):

    mix_num = 5  # default
    no_multichannel_lmm = False

    def __init__(
        self,
        ep_params,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        mean, log_sigma, autoregression_coefficients, mixture_weights = torch.split(ep_params, self.mix_num * 3, dim=1)

        N, C, H, W = mean.shape
        self.mean = torch.reshape(mean, (N, 3, self.mix_num, H, W))

        self.log_sigma = torch.reshape(log_sigma, (N, 3, self.mix_num, H, W))
        self.log_sigma = torch.clamp(self.log_sigma, min=-7.0)

        # log weights
        if self.no_multichannel_lmm:
            self.mixture_weights = torch.reshape(mixture_weights, (N, 1, self.mix_num, H, W)) + torch.zeros(
                N, 3, self.mix_num, H, W, device=mixture_weights.device
            )
        else:
            self.mixture_weights = torch.reshape(mixture_weights, (N, 3, self.mix_num, H, W))

        self.coeffs = torch.tanh(autoregression_coefficients)
        self.coeffs = torch.reshape(self.coeffs, (N, 3, self.mix_num, H, W))

    def forward(self, input, bit_depth):
        N, C, H, W = input.shape

        half = float(0.5) / (2**bit_depth - 1) * 2  # B C H W
        half = half.unsqueeze(2).repeat(1, 1, self.mix_num, 1, 1)

        x = torch.reshape(input, (N, C, 1, H, W)) + torch.zeros(N, C, self.mix_num, H, W, device=input.device)

        m1 = torch.reshape(self.mean[:, 0, :, :, :], (N, 1, self.mix_num, H, W))
        m2 = self.mean[:, 1, :, :, :] + self.coeffs[:, 0, :, :, :] * x[:, 0, :, :, :]
        m2 = torch.reshape(m2, (N, 1, self.mix_num, H, W))
        m3 = (
            self.mean[:, 2, :, :, :]
            + self.coeffs[:, 1, :, :, :] * x[:, 0, :, :, :]
            + self.coeffs[:, 2, :, :, :] * x[:, 1, :, :, :]
        )
        m3 = torch.reshape(m3, (N, 1, self.mix_num, H, W))
        self.mean = torch.cat((m1, m2, m3), 1)

        centered_x = x - self.mean
        inv_sigma = torch.exp(-self.log_sigma)
        plus_in = inv_sigma * (centered_x + half)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_sigma * (centered_x - half)
        cdf_min = torch.sigmoid(min_in)
        log_one_minus_cdf_min = -F.softplus(min_in)  # 255
        log_cdf_plus = plus_in - F.softplus(plus_in)  # -255
        cdf_delta = cdf_plus - cdf_min

        log_probs = torch.where(
            x - half < 0.00001,
            log_cdf_plus,
            torch.where(x + half > 1.99999, log_one_minus_cdf_min, torch.log(torch.clamp(cdf_delta, min=1e-9))),
        )
        log_probs = log_probs + self.log_prob_from_logits(self.mixture_weights)

        return self.log_sum_exp(log_probs)

    def log_prob_from_logits(self, x):  # normalize
        axis = 2
        m = torch.amax(x, axis, keepdim=True)
        return x - m - torch.log(torch.sum(torch.exp(x - m), axis, keepdim=True))

    def log_sum_exp(self, x):
        axis = 2
        m = torch.amax(x, axis)
        m2 = torch.amax(x, axis, keepdim=True)

        return m + torch.log(torch.sum(torch.exp(x - m2), axis))


class MultiRGBMixtureLogistic(nn.Module):

    mix_num = 5  # default
    no_multichannel_lmm = False

    def __init__(
        self,
        ep_params,  # 60 ->
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        ep_ch = 10 * self.mix_num if self.no_multichannel_lmm else 12 * self.mix_num
        num_rgb = ep_params.shape[1] // ep_ch
        self.num_rgb = num_rgb
        self.means = []
        self.log_sigmas = []
        self.mixture_weights = []
        self.coeffs = []
        for i in range(num_rgb):
            start = i * ep_ch
            end = (i + 1) * ep_ch
            mean, log_sigma, autoregression_coefficients, mixture_weights = torch.split(
                ep_params[:, start:end, :, :], self.mix_num * 3, dim=1
            )

            N, C, H, W = mean.shape
            self.means.append(torch.reshape(mean, (N, 3, self.mix_num, H, W)))

            self.log_sigmas.append(torch.reshape(log_sigma, (N, 3, self.mix_num, H, W)))
            self.log_sigmas[-1] = torch.clamp(self.log_sigmas[-1], min=-7.0)

            # log weights
            if self.no_multichannel_lmm:
                self.mixture_weights.append(
                    torch.reshape(mixture_weights, (N, 1, self.mix_num, H, W))
                    + torch.zeros(N, 3, self.mix_num, H, W, device=mixture_weights.device)
                )
            else:
                self.mixture_weights.append(torch.reshape(mixture_weights, (N, 3, self.mix_num, H, W)))

            self.coeffs.append(torch.tanh(autoregression_coefficients))
            self.coeffs[-1] = torch.reshape(self.coeffs[-1], (N, 3, self.mix_num, H, W))

    def forward(self, inputs):
        N, _, H, W = inputs.shape
        half = float(0.5) / 255.0 * 2

        log_probss = torch.zeros_like(inputs, device=inputs.device)

        inputs = torch.chunk(inputs, self.num_rgb, dim=1)
        for idx, x in enumerate(inputs):
            x = torch.reshape(x, (N, 3, 1, H, W)) + torch.zeros(N, 3, self.mix_num, H, W, device=x.device)
            mean = self.means.pop(0)
            log_sigma = self.log_sigmas.pop(0)
            mixture_weights = self.mixture_weights.pop(0)
            coeffs = self.coeffs.pop(0)

            m1 = torch.reshape(mean[:, 0, :, :, :], (N, 1, self.mix_num, H, W))
            m2 = mean[:, 1, :, :, :] + coeffs[:, 0, :, :, :] * x[:, 0, :, :, :]
            m2 = torch.reshape(m2, (N, 1, self.mix_num, H, W))
            m3 = (
                mean[:, 2, :, :, :]
                + coeffs[:, 1, :, :, :] * x[:, 0, :, :, :]
                + coeffs[:, 2, :, :, :] * x[:, 1, :, :, :]
            )
            m3 = torch.reshape(m3, (N, 1, self.mix_num, H, W))
            mean = torch.cat((m1, m2, m3), 1)

            centered_x = x - mean
            inv_sigma = torch.exp(-log_sigma)
            plus_in = inv_sigma * (centered_x + half)
            cdf_plus = torch.sigmoid(plus_in)
            min_in = inv_sigma * (centered_x - half)
            cdf_min = torch.sigmoid(min_in)
            log_one_minus_cdf_min = -F.softplus(min_in)  # 255
            log_cdf_plus = plus_in - F.softplus(plus_in)  # -255
            cdf_delta = cdf_plus - cdf_min

            log_probs = torch.where(
                x - half < 0.001,
                log_cdf_plus,
                torch.where(x + half > 1.999, log_one_minus_cdf_min, torch.log(torch.clamp(cdf_delta, min=1e-9))),
            )
            log_probs = log_probs + self.log_prob_from_logits(mixture_weights)

            log_probss[:, 3 * idx : 3 * (idx + 1), :, :] = self.log_sum_exp(log_probs)
        return log_probss

    def log_prob_from_logits(self, x):  # normalize
        axis = 2
        m = torch.amax(x, axis, keepdim=True)
        return x - m - torch.log(torch.sum(torch.exp(x - m), axis, keepdim=True))

    def log_sum_exp(self, x):
        axis = 2
        m = torch.amax(x, axis)
        m2 = torch.amax(x, axis, keepdim=True)

        return m + torch.log(torch.sum(torch.exp(x - m2), axis))
