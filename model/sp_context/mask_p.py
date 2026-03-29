from compressai.layers import MaskedConv2d
from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d_P(MaskedConv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, mask_type=mask_type, **kwargs)
        # self.mask[:, :, 2, 5:7] = 0
        self.mask[:, :, 1, 4] = 0

    def forward(self, x, *args, **kwargs):
        return super().forward(x)


class MaskedConv2d_P_64(MaskedConv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, mask_type=mask_type, **kwargs)
        self.mask[:, :, 2, 5:7] = 0
        # self.mask[:, :, 1, 4] = 0

    def forward(self, x, *args, **kwargs):
        return super().forward(x)

    @staticmethod
    def get_coding_table(patch_sz=64):
        COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)
        for i in range(patch_sz):
            start = 2 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
        return COT


class CondMaskedConv2d_P_64(MaskedConv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, mask_type=mask_type, **kwargs)
        self.mask[:, :, 2, 5:7] = 0
        # self.mask[:, :, 1, 4] = 0

    def forward(self, x, cond, *args, **kwargs):

        return super().forward(torch.cat([F.interpolate(cond, size=x.shape[2:]), x], dim=1))

    @staticmethod
    def get_coding_table(patch_sz=64):
        COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)
        for i in range(patch_sz):
            start = 2 * i + 1
            COT[i, :] = torch.arange(start, start + patch_sz)
        return COT


class CondMaskedConv2d_P_column(MaskedConv2d):
    def __init__(self, *args: Any, mask_type: str = "A", **kwargs: Any):
        super().__init__(*args, mask_type=mask_type, **kwargs)
        self.mask[:, :, :, 3:7] = 0
        self.mask[:, :, :, 0:3] = 1

    def forward(self, x, cond, *args, **kwargs):

        return super().forward(torch.cat([F.interpolate(cond, size=x.shape[2:]), x], dim=1))

    @staticmethod
    def get_coding_table(patch_sz=64):
        COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)
        for i in range(patch_sz):
            COT[i, :] = torch.arange(1, 1 + patch_sz)
        return COT


class MultiMaskedConv2d_P(nn.Module):
    def __init__(self, in_ch_list, out_ch_list, mask_type="A", **layer_kwargs):
        super().__init__()
        self.partite = len(in_ch_list)
        self.ctx = nn.ModuleDict(
            {
                f"sp_group_{g}": MaskedConv2d_P(
                    in_ch_list[g],
                    out_ch_list[g],
                    mask_type=mask_type,
                    **layer_kwargs,
                )
                for g in range(len(in_ch_list))
            }
        )

    def forward(self, x, idx):
        out = self.ctx[f"sp_group_{idx}"](x[idx])
        return out

    def init_activate_channels(self, *args, **kwargs):
        pass

    @staticmethod
    def get_coding_table(block_sz=16):
        COT = torch.zeros(block_sz, block_sz, dtype=torch.int64)
        for i in range(block_sz):
            start = 2 * i + 1
            COT[i, :] = torch.arange(start, start + block_sz)
        return COT


class MultiMaskedConv2d_P_64(nn.Module):
    def __init__(self, in_ch_list, out_ch_list, mask_type="A", **layer_kwargs):
        super().__init__()
        self.partite = len(in_ch_list)
        self.ctx = nn.ModuleDict(
            {
                f"sp_group_{g}": MaskedConv2d_P_64(
                    in_ch_list[g],
                    out_ch_list[g],
                    mask_type=mask_type,
                    **layer_kwargs,
                )
                for g in range(len(in_ch_list))
            }
        )

    def forward(self, x, idx):
        out = self.ctx[f"sp_group_{idx}"](x[idx])
        return out

    def init_activate_channels(self, *args, **kwargs):
        pass

    @staticmethod
    def get_coding_table(block_sz=16):
        COT = torch.zeros(block_sz, block_sz, dtype=torch.int64)
        for i in range(block_sz):
            start = 2 * i + 1
            COT[i, :] = torch.arange(start, start + block_sz)
        return COT
