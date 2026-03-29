import torch.nn as nn
from compressai.models import CompressionModel
from compressai.latent_codecs import LatentCodec
import torch
import torch.nn.functional as F
import torchac
import time

mix_num = 5
mix_num2 = mix_num * 2
mix_num3 = mix_num * 3


class EncWrapper(nn.Module):
    def __init__(self, codec, chunk_sz=10000):

        super().__init__()
        self.codec = codec
        self.chunk_sz = chunk_sz

    def forward(self, x, jpg_patch):

        B = x.shape[0]
        x_stream = []
        latent_code, bit_depth = self.codec.compress_latent(x, jpg_patch)
        device = x.device
        prior_total = self.codec.decompress_latent(**latent_code)["prior"]
        norm_scale = (1 / (2**bit_depth - 1) * 2).reshape(-1, 1, 1, 1)  # (B,1,1,1)
        half = 0.5 * norm_scale

        max_width = 2**self.codec.end_bit
        idx = torch.arange(max_width, device=device).unsqueeze(0)
        valid_pos = idx < (2**bit_depth).unsqueeze(1)  # (B, max_width)
        samples = idx * valid_pos

        COT = self.codec.sp_ctx.get_coding_table()
        max_step = torch.max(COT)
        max_num_pixels = torch.bincount(COT.flatten()).max()

        samples = samples * norm_scale.reshape(-1, 1)  # (B, max_width)
        samples = samples.reshape(B, 1, 1, -1).repeat(1, max_num_pixels, mix_num, 1)
        valid_pos = valid_pos.reshape(B, 1, 1, -1).repeat(1, max_num_pixels, mix_num, 1)

        # with misc.Timer(results, "x compress"):
        context_total = self.codec.sp_ctx(x * norm_scale, jpg_patch)

        for i in range(max_step):
            # t1 = time.time()
            h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
            # print(i, len(h_idx))
            context = context_total[:, :, h_idx, w_idx].unsqueeze(3)
            prior = prior_total[:, :, h_idx, w_idx].unsqueeze(3)
            x_crop = x[:, :, h_idx, w_idx].unsqueeze(3)

            fusion_context = self.codec.fusion(torch.cat([prior, context], dim=1))
            ep_params = self.codec.ep(fusion_context, jpg_patch)
            mu, log_sigma, coeffs, weights = torch.split(ep_params, 3 * mix_num, dim=1)
            if self.codec.distribution.no_multichannel_lmm:
                weights = weights.reshape(B, 1, mix_num, -1, 1)
                weights = weights.repeat(1, 3, 1, 1, 1)
            else:
                weights = weights.reshape(B, 3, mix_num, -1, 1)
            coeffs = torch.tanh(coeffs)

            for c in range(3):
                if c == 0:
                    mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)

                elif c == 1:
                    mu_c = mu[:, mix_num:mix_num2, :, :] + (x_crop[:, 0:1, :] * norm_scale) * coeffs[:, :mix_num, :, :]
                    mu_c = mu_c.permute(0, 2, 1, 3)

                elif c == 2:
                    mu_c = (
                        mu[:, mix_num2:, :, :]
                        + (x_crop[:, 0:1, :] * norm_scale) * coeffs[:, mix_num:mix_num2, :, :]
                        + (x_crop[:, 1:2, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                    )
                    mu_c = mu_c.permute(0, 2, 1, 3)

                # samples_c = samples.repeat(1, mu_c.shape[1], 1, 1)
                # valid_pos_c = valid_pos.repeat(1, mu_c.shape[1], 1, 1)
                samples_c = samples[:, : mu_c.shape[1], :, :]
                valid_pos_c = valid_pos[:, : mu_c.shape[1], :, :]

                samples_centered = samples_c - mu_c
                inv_sigma = torch.exp(-log_sigma[:, c * mix_num : (c + 1) * mix_num, :, :].permute(0, 2, 1, 3))
                plus_in = inv_sigma * (samples_centered + half)
                cdf_plus = torch.sigmoid(plus_in)
                min_in = inv_sigma * (samples_centered - half)
                cdf_min = torch.sigmoid(min_in)
                cdf_delta = cdf_plus - cdf_min
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))

                cdf_delta = torch.where(
                    samples_c - half < 0.00001,
                    cdf_plus,
                    torch.where(samples_c + half > 1.99999, one_minus_cdf_min, cdf_delta),
                )

                cdf_delta = cdf_delta * valid_pos_c

                weights_c = weights[:, c, :, :, :].permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1.0 / 64800, 1.0)
                pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0.0, 1.0)
                cdf = F.pad(cdf, (1, 0))
                symbol = x_crop[:, c].short().reshape(B, -1)
                stream = torchac.encode_float_cdf(
                    cdf.cpu(), symbol.cpu(), needs_normalization=False, check_input_bounds=False
                )
                x_stream.append(stream)
            # t2 = time.time()
            # print(f"Step {i} time: {t2 - t1} seconds")
        return (latent_code, x_stream, bit_depth)

    def get_bit_depth_num(self):
        return self.codec.get_bit_depth_num()


class DecWrapper(nn.Module):
    def __init__(self, codec, chunk_sz=10000):
        super().__init__()
        self.codec = codec
        self.chunk_sz = chunk_sz

    def forward(self, latent_code, x_stream, bit_depth, jpg_patch):

        prior_total = self.codec.decompress_latent(**latent_code)["prior"]
        device = prior_total.device
        B = prior_total.shape[0]

        norm_scale = (1 / (2**bit_depth - 1) * 2).reshape(-1, 1, 1, 1)  # (B,1,1,1)
        half = 0.5 * norm_scale

        max_width = 2**self.codec.end_bit
        idx = torch.arange(max_width, device=bit_depth.device).unsqueeze(0)  # (1, max_width)
        valid_pos = idx < (2**bit_depth).unsqueeze(1)  # (B, max_width)
        samples = idx * valid_pos

        COT = self.codec.sp_ctx.get_coding_table()
        max_step = torch.max(COT)
        max_num_pixels = torch.bincount(COT.flatten()).max()

        samples = samples * norm_scale.reshape(-1, 1)  # (B, max_width)
        samples = samples.reshape(B, 1, 1, -1).repeat(1, max_num_pixels, mix_num, 1)
        valid_pos = valid_pos.reshape(B, 1, 1, -1).repeat(1, max_num_pixels, mix_num, 1)

        # with misc.Timer(results, "x decompress"):
        x_tmp = torch.zeros(prior_total.shape[0], 3, prior_total.shape[2], prior_total.shape[3], device=device)

        j = 0
        for i in range(max_step):
            # print(i)
            h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
            context = self.codec.sp_ctx(x_tmp * norm_scale, jpg_patch)[:, :, h_idx, w_idx].unsqueeze(3)
            prior = prior_total[:, :, h_idx, w_idx].unsqueeze(3)
            x_crop = x_tmp[:, :, h_idx, w_idx].unsqueeze(3)
            fusion_context = self.codec.fusion(torch.cat([prior, context], dim=1))
            ep_params = self.codec.ep(fusion_context, jpg_patch)
            mu, log_sigma, coeffs, weights = torch.split(ep_params, 3 * mix_num, dim=1)
            if self.codec.distribution.no_multichannel_lmm:
                weights = weights.reshape(weights.shape[0], 1, mix_num, -1, 1)
                weights = weights.repeat(1, 3, 1, 1, 1)
            else:

                weights = weights.reshape(weights.shape[0], 3, mix_num, -1, 1)
            coeffs = torch.tanh(coeffs)
            for c in range(3):
                if c == 0:
                    mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)

                elif c == 1:
                    mu_c = mu[:, mix_num:mix_num2, :, :] + (x_crop[:, 0:1, :] * norm_scale) * coeffs[:, :mix_num, :, :]
                    mu_c = mu_c.permute(0, 2, 1, 3)

                else:
                    mu_c = (
                        mu[:, mix_num2:, :, :]
                        + (x_crop[:, 0:1, :, :] * norm_scale) * coeffs[:, mix_num:mix_num2, :, :]
                        + (x_crop[:, 1:2, :, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                    )
                    mu_c = mu_c.permute(0, 2, 1, 3)

                # samples_c = samples.repeat(1, mu_c.shape[1], 1, 1)
                # valid_pos_c = valid_pos.repeat(1, mu_c.shape[1], 1, 1)
                samples_c = samples[:, : mu_c.shape[1], :, :]
                valid_pos_c = valid_pos[:, : mu_c.shape[1], :, :]

                samples_centered = samples_c - mu_c
                inv_sigma = torch.exp(-log_sigma[:, c * mix_num : (c + 1) * mix_num, :, :].permute(0, 2, 1, 3))
                plus_in = inv_sigma * (samples_centered + half)
                cdf_plus = torch.sigmoid(plus_in)
                min_in = inv_sigma * (samples_centered - half)
                cdf_min = torch.sigmoid(min_in)
                cdf_delta = cdf_plus - cdf_min
                one_minus_cdf_min = torch.exp(-F.softplus(min_in))
                cdf_plus = torch.exp(plus_in - F.softplus(plus_in))

                cdf_delta = torch.where(
                    samples_c - half < 0.00001,
                    cdf_plus,
                    torch.where(samples_c + half > 1.99999, one_minus_cdf_min, cdf_delta),
                )

                cdf_delta = cdf_delta * valid_pos_c

                weights_c = weights[:, c, :, :, :].permute(0, 2, 1, 3)
                m = torch.amax(weights_c, 2, keepdim=True)
                weights_c = torch.exp(weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True)))
                pmf = torch.sum(cdf_delta * weights_c, dim=2)

                pmf = pmf.clamp_(1 / 64800, 1.0)
                pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                cdf = torch.cumsum(pmf, dim=2).clamp_(0.0, 1.0)
                cdf = F.pad(cdf, (1, 0))

                symbol_out = torchac.decode_float_cdf(cdf.cpu(), x_stream[j], needs_normalization=False)

                x_crop[:, c, :, 0] = symbol_out.float().reshape(B, -1)
                j += 1

            x_tmp[:, :, h_idx, w_idx] = x_crop.squeeze(3)
        return x_tmp

    def get_bit_depth_num(self):
        return self.codec.get_bit_depth_num()
