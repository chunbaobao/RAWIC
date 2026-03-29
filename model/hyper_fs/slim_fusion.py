import torch
import torch.nn as nn
import numpy as np
from model.slim import SlimHeadConv2d, SlimTailConv2d


class SlimFusion(nn.Module):
    def __init__(
        self,
        num_k,
        in_ch: int,  # 48
        out_ch_per_ch: int,  # 32
        mid_ch: int = None,  #
        num_layers=3,
        make_layer=None,
        make_act=None,
        skip_last_act: bool = True,
        **layer_kwargs,
    ):
        super().__init__()

        self.out_ch = num_k * out_ch_per_ch
        self.out_ch_per_ch = out_ch_per_ch
        if mid_ch is None:
            mid_ch = self.out_ch

        tail = SlimTailConv2d(mid_ch, self.out_ch, **layer_kwargs)

        layers = []
        for idx in range(num_layers):
            if idx == 0:
                layers.append(
                    make_layer(
                        in_ch,
                        mid_ch,
                        **layer_kwargs,
                    )
                )

            elif idx == num_layers - 1:
                layers.append(tail)
            else:
                layers.append(
                    make_layer(
                        mid_ch,
                        mid_ch,
                        **layer_kwargs,
                    )
                )
            if not (skip_last_act and idx == num_layers - 1) and make_act is not None:
                layers.append(make_act())

        self.fusion = nn.Sequential(*layers)

    def forward(self, x, idx):
        self.set_activate_channels(idx)
        out = self.fusion(x)

        return out

    def set_activate_channels(self, idx):
        for m in self.fusion.modules():
            if isinstance(m, SlimTailConv2d):
                m.set_activate_channels(
                    self.sample_idx[idx] * self.out_ch_per_ch, self.sample_idx[idx + 1] * self.out_ch_per_ch
                )

    def init_activate_channels(self, sample_ch: list):
        sample_ch = np.array(sample_ch)
        sample_idx = np.pad(sample_ch.cumsum(), (1, 0)).astype(int)
        self.sample_idx = sample_idx
