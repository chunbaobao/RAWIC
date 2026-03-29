import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class CustomEntropyBottleneck(EntropyBottleneck):
    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        B, C, H, W = x.shape
        return super(EntropyBottleneck, self).compress(
            x.reshape(1, B * C, H, W), indexes.reshape(1, B * C, H, W), medians.reshape(1, B * C, 1, 1)
        )

    def decompress(self, strings, size, BC):
        output_size = (BC[0], self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(BC[0], *([-1] * (len(size) + 1)))

        z_hat = super(EntropyBottleneck, self).decompress(
            strings, indexes.reshape(1, BC[0] * BC[1], *size), torch.float, medians.reshape(1, BC[0] * BC[1], 1, 1)
        )
        return z_hat.reshape(BC[0], BC[1], size[0], size[1])
