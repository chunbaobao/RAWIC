# import torch.nn as nn
# from model.custom_layers import ResBlock_1x1_ds


# class MultiResidualEntropyModel(nn.Module):
#     def __init__(self, in_ch_list: list, out_ch_list: list):
#         super().__init__()
#         self.in_ch_list = in_ch_list
#         self.out_ch_list = out_ch_list
#         self.ep = nn.ModuleDict(
#             {f"ep_group_{g}": ResBlock_1x1_ds(in_ch_list[g], out_ch_list[g]) for g in range(len(out_ch_list))}
#         )

#     def forward(self, x, idx):
#         out = self.ep[f"ep_group_{idx}"](x)
#         return out

#     def init_activate_channels(self, *args, **kwargs):
#         pass


import torch.nn as nn
from compressai.layers import conv1x1
from model.custom_layers import ResBlock_1x1
from model.base import JpegConditionedSequential


class ResidualEntropyModel(nn.Module):
    def __init__(self, ep_ch_in, ep_ch_out):

        super().__init__()

        # The output conv layers for each class
        self.conv_outs = nn.Sequential(
            conv1x1(ep_ch_in, ep_ch_in),
            ResBlock_1x1(ep_ch_in),
            conv1x1(ep_ch_in, ep_ch_in),  # ?
            nn.LeakyReLU(inplace=True),
            conv1x1(ep_ch_in, ep_ch_out),  # ep_ch_out represents the parameters of the distribution
        )

    def forward(self, fusion_context, *args, **kwargs):
        out = self.conv_outs(fusion_context)
        return out


class ResidualEntropyModel_v2(nn.Module):
    def __init__(self, ep_ch_in, ep_ch_out):

        super().__init__()

        # The output conv layers for each class
        self.conv_outs = nn.Sequential(
            conv1x1(ep_ch_in, ep_ch_in),
            ResBlock_1x1(ep_ch_in),
            conv1x1(ep_ch_in, ep_ch_in),  # ?
            nn.LeakyReLU(inplace=True),
            conv1x1(ep_ch_in, ep_ch_out),  # ep_ch_out represents the parameters of the distribution
        )

    def forward(self, fusion_context):
        out = self.conv_outs(fusion_context)
        return out


class MultiResidualEntropyModel(nn.Module):
    def __init__(self, in_ch_list: list, out_ch_list: list):
        super().__init__()
        self.in_ch_list = in_ch_list
        self.out_ch_list = out_ch_list
        self.ep = nn.ModuleDict(
            {f"ep_group_{g}": ResidualEntropyModel(in_ch_list[g], out_ch_list[g]) for g in range(len(out_ch_list))}
        )

    def forward(self, x, idx):
        out = self.ep[f"ep_group_{idx}"](x)
        return out

    def init_activate_channels(self, *args, **kwargs):
        pass


class CondResidualEntropyModel(nn.Module):
    def __init__(self, ep_ch_in, ep_ch_out):

        super().__init__()

        # The output conv layers for each class
        self.conv_outs = JpegConditionedSequential(
            conv1x1(ep_ch_in + 3, ep_ch_in),
            ResBlock_1x1(ep_ch_in + 3),
            conv1x1(ep_ch_in + 6, ep_ch_in),  # ?
            nn.LeakyReLU(inplace=True),
            conv1x1(ep_ch_in + 3, ep_ch_out),  # ep_ch_out represents the parameters of the distribution
        )

    def forward(self, fusion_context, x_rgb):
        out = self.conv_outs(fusion_context, x_rgb)
        return out
