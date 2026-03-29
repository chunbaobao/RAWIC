import torch
import torch.nn as nn
from compressai.layers import sequential_channel_ramp


class MultiChannelContext(nn.Module):
    def __init__(
        self,
        in_ch_list: list,
        out_ch_list: list,
        min_ch=0,
        num_layers=3,
        method="linear",
        make_layer=None,
        make_act=None,
        skip_last_act: bool = True,
        **layer_kwargs,
    ):

        super().__init__()
        self.out_ch_list = out_ch_list
        self.ctx = nn.ModuleDict(
            {
                f"ch_group_{g}": sequential_channel_ramp(
                    in_ch=sum(in_ch_list[:g]),
                    out_ch=out_ch_list[g],
                    min_ch=min_ch,
                    num_layers=num_layers,
                    interp=method,
                    make_layer=make_layer,
                    make_act=make_act,
                    skip_last_act=skip_last_act,
                    **layer_kwargs,
                )
                for g in range(1, len(in_ch_list))
            }
        )

    # def forward(self, x, idx):
    #     if idx == 0:
    #         return x[0].new_empty(0)
    #     x_group = self.merge(*x[:idx])
    #     out = self.ctx[f"ch_group_{idx}"](x_group)
    #     return out

    def forward(self, x, idx):
        if idx == 0:
            b = x[0].shape[0]
            c = self.out_ch_list[0]
            h, w = x[0].shape[2], x[0].shape[3]
            return torch.zeros(b, c, h, w).to(x[0].device)
        x_group = self.merge(*x[:idx])
        out = self.ctx[f"ch_group_{idx}"](x_group)
        return out

    def merge(self, *args):
        if len(args) == 0:
            return torch.Tensor()
        return torch.cat(args, dim=1)

    def init_activate_channels(self, *args, **kwargs):
        pass
