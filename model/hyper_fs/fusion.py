import torch
import torch.nn as nn
import math


def ramp(a, b, steps=None, method="linear", **kwargs):
    if method == "linear":
        return torch.linspace(a, b, steps, **kwargs)
    if method == "log":
        return torch.logspace(math.log10(a), math.log10(b), steps, **kwargs)
    raise ValueError(f"Unknown ramp method: {method}")


def sequential_channel_ramp(
    in_ch: int,
    out_ch: int,
    *,
    min_ch: int = 0,
    num_layers: int = 3,
    interp: str = "linear",
    make_layer=None,
    make_act=None,
    skip_last_act: bool = True,
    **layer_kwargs,
) -> nn.Module:
    """Interleave layers of gradually ramping channels with nonlinearities."""
    channels = ramp(in_ch, out_ch, num_layers + 1, method=interp).floor().int()
    channels[1:-1] = channels[1:-1].clip(min=min_ch)
    channels[0] = in_ch
    channels[-1] = out_ch
    channels = channels.tolist()
    layers = [
        module
        for ch_in, ch_out in zip(channels[:-1], channels[1:])
        for module in [
            make_layer(ch_in, ch_out, **layer_kwargs),
            make_act(),
        ]
    ]
    if skip_last_act:
        layers = layers[:-1]
    return nn.Sequential(*layers)


class MultiFusion(nn.Module):
    def __init__(
        self,
        in_ch: int,
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
        self.fusion = nn.ModuleDict(
            {
                f"fusion_{g}": sequential_channel_ramp(
                    in_ch=in_ch,
                    out_ch=out_ch_list[g],
                    min_ch=min_ch,
                    num_layers=num_layers,
                    interp=method,
                    make_layer=make_layer,
                    make_act=make_act,
                    skip_last_act=skip_last_act,
                    **layer_kwargs,
                )
                for g in range(len(out_ch_list))
            }
        )

    def forward(self, x, idx):
        out = self.fusion[f"fusion_{idx}"](x)

        return out

    def init_activate_channels(self, *args, **kwargs):
        pass
