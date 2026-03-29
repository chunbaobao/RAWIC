import torch
import torch.nn as nn
from compressai.latent_codecs import LatentCodec
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping
from torch import Tensor
from compressai.ops import quantize_ste
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import CheckerboardMaskedConv2d
from compressai.entropy_models import EntropyModel
from itertools import accumulate


class CustomHyperLatentCodec(LatentCodec):
    """Entropy bottleneck codec with surrounding `h_a` and `h_s` transforms.

    "Hyper" side-information branch introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: ``HyperLatentCodec`` should be used inside
       ``HyperpriorLatentCodec`` to construct a full hyperprior.

    .. code-block:: none

               ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
        y ──►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►── params
               └───┘     └───┘        EB        └───┘

    """

    entropy_bottleneck: EntropyBottleneck
    h_a: nn.Module
    h_s: nn.Module

    def __init__(
        self,
        entropy_bottleneck: Optional[EntropyBottleneck] = None,
        h_a: Optional[nn.Module] = None,
        h_s: Optional[nn.Module] = None,
        quantizer: str = "noise",
        **kwargs,
    ):
        super().__init__()
        assert entropy_bottleneck is not None
        self.entropy_bottleneck = entropy_bottleneck
        self.h_a = h_a or nn.Identity()
        self.h_s = h_s or nn.Identity()
        self.quantizer = quantizer

    def forward(self, y: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.quantizer == "ste":
            z_medians = self.entropy_bottleneck._get_medians()
            z_hat = quantize_ste(z - z_medians) + z_medians
        params = self.h_s(z_hat)
        return {"likelihoods": {"z": z_likelihoods}, "params": params}

    def compress(self, y: Tensor) -> Dict[str, Any]:
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], z.size()[:2])
        params = self.h_s(z_hat)
        return {"strings": [z_strings], "shape": z.size(), "params": params}

    def decompress(self, strings, shape, **kwargs) -> Dict[str, Any]:
        (z_strings,) = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape[2:], shape[:2])
        params = self.h_s(z_hat)
        return {"params": params}


class CondHyperLatentCodec(LatentCodec):

    entropy_bottleneck: EntropyBottleneck
    h_a: nn.Module
    h_s: nn.Module

    def __init__(
        self,
        entropy_bottleneck: Optional[EntropyBottleneck] = None,
        h_a: Optional[nn.Module] = None,
        h_s: Optional[nn.Module] = None,
        quantizer: str = "noise",
        **kwargs,
    ):
        super().__init__()
        assert entropy_bottleneck is not None
        self.entropy_bottleneck = entropy_bottleneck
        self.h_a = h_a or nn.Identity()
        self.h_s = h_s or nn.Identity()
        self.quantizer = quantizer

    def forward(self, y: Tensor, cond: Tensor) -> Dict[str, Any]:
        z = self.h_a(y, cond)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.quantizer == "ste":
            z_medians = self.entropy_bottleneck._get_medians()
            z_hat = quantize_ste(z - z_medians) + z_medians
        params = self.h_s(z_hat, cond)
        return {"likelihoods": {"z": z_likelihoods}, "params": params}

    def compress(self, y: Tensor, cond: Tensor) -> Dict[str, Any]:
        z = self.h_a(y, cond)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], z.size()[:2])
        params = self.h_s(z_hat, cond)
        return {"strings": [z_strings], "shape": z.size(), "params": params}

    def decompress(self, strings, shape, cond, **kwargs) -> Dict[str, Any]:
        (z_strings,) = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape[2:], shape[:2])
        params = self.h_s(z_hat, cond)
        return {"params": params}


class CustomGaussianConditionalLatentCodec(LatentCodec):
    """Gaussian conditional for compressing latent ``y`` using ``ctx_params``.

    Probability model for Gaussian of ``(scales, means)``.

    Gaussian conditonal entropy model introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. note:: Unlike the original paper, which models only the scale
       (i.e. "width") of the Gaussian, this implementation models both
       the scale and the mean (i.e. "center") of the Gaussian.

    .. code-block:: none

                          ctx_params
                              │
                              ▼
                              │
                           ┌──┴──┐
                           │  EP │
                           └──┬──┘
                              │
               ┌───┐  y_hat   ▼
        y ──►──┤ Q ├────►────····──►── y_hat
               └───┘          GC

    """

    gaussian_conditional: GaussianConditional
    entropy_parameters: nn.Module

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]] = None,
        gaussian_conditional: Optional[GaussianConditional] = None,
        entropy_parameters: Optional[nn.Module] = None,
        quantizer: str = "noise",
        chunks: Tuple[str] = ("scales", "means"),
        **kwargs,
    ):
        super().__init__()
        self.quantizer = quantizer
        self.gaussian_conditional = gaussian_conditional or GaussianConditional(scale_table, **kwargs)
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.chunks = tuple(chunks)

    def forward(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        if self.quantizer == "ste":
            y_hat = quantize_ste(y - means_hat) + means_hat
        return {"likelihoods": {"y": y_likelihoods}, "y_hat": y_hat}

    def compress(self, y: Tensor, ctx_params: Tensor) -> Dict[str, Any]:
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)
        B, C, H, W = y.shape
        indexes = self.gaussian_conditional.build_indexes(scales_hat.reshape(1, B * C, H, W))
        y_strings = self.gaussian_conditional.compress(
            y.reshape(1, B * C, H, W), indexes, means_hat.reshape(1, B * C, H, W)
        )
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat.reshape(1, B * C, H, W))
        y_hat = y_hat.reshape(B, C, H, W)
        return {"strings": [y_strings], "shape": y.shape[2:4], "y_hat": y_hat}
        # return {"strings": [y_strings], "shape": y.shape[2:4]}

    def decompress(
        self,
        strings: List[List[bytes]],
        ctx_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        (y_strings,) = strings
        gaussian_params = self.entropy_parameters(ctx_params)
        scales_hat, means_hat = self._chunk(gaussian_params)

        B, C, H, W = means_hat.shape

        indexes = self.gaussian_conditional.build_indexes(scales_hat.reshape(1, B * C, H, W))
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat.reshape(1, B * C, H, W))
        y_hat = y_hat.reshape(B, C, H, W)
        return {"y_hat": y_hat}

    def _chunk(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        scales, means = None, None
        if self.chunks == ("scales",):
            scales = params
        if self.chunks == ("means",):
            means = params
        if self.chunks == ("scales", "means"):
            scales, means = params.chunk(2, 1)
        if self.chunks == ("means", "scales"):
            means, scales = params.chunk(2, 1)
        return scales, means


class CustomHyperpriorLatentCodec(LatentCodec):
    """Hyperprior codec constructed from latent codec for ``y`` that
    compresses ``y`` using ``params`` from ``hyper`` branch.

    Hyperprior entropy modeling introduced in
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_,
    by J. Balle, D. Minnen, S. Singh, S.J. Hwang, and N. Johnston,
    International Conference on Learning Representations (ICLR), 2018.

    .. code-block:: none

                 ┌──────────┐
            ┌─►──┤ lc_hyper ├──►─┐
            │    └──────────┘    │
            │                    ▼ params
            │                    │
            │                 ┌──┴───┐
        y ──┴───────►─────────┤ lc_y ├───►── y_hat
                              └──────┘

    By default, the following codec is constructed:

    .. code-block:: none

                 ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            ┌─►──┤h_a├──►──┤ Q ├───►───····───►───┤h_s├──►─┐
            │    └───┘     └───┘        EB        └───┘    │
            │                                              │
            │                  ┌──────────────◄────────────┘
            │                  │            params
            │               ┌──┴──┐
            │               │  EP │
            │               └──┬──┘
            │                  │
            │   ┌───┐  y_hat   ▼
        y ──┴─►─┤ Q ├────►────····────►── y_hat
                └───┘          GC

    Common configurations of latent codecs include:
     - entropy bottleneck ``hyper`` (default) and gaussian conditional ``y`` (default)
     - entropy bottleneck ``hyper`` (default) and autoregressive ``y``
    """

    latent_codec: Mapping[str, LatentCodec]

    def __init__(self, latent_codec: Optional[Mapping[str, LatentCodec]] = None, **kwargs):
        super().__init__()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": CustomGaussianConditionalLatentCodec,
                "hyper": CustomHyperLatentCodec,
            },
            save_direct=True,
        )

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"](y)
        y_out = self.latent_codec["y"](y, hyper_out["params"])
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
            "y_hat": y_out["y_hat"],
        }

    def compress(self, y: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"].compress(y)
        y_out = self.latent_codec["y"].compress(y, hyper_out["params"])
        [z_strings] = hyper_out["strings"]
        return {
            "strings": [*y_out["strings"], z_strings],
            "shape": {"y": y_out["shape"], "hyper": hyper_out["shape"]},
            "y_hat": y_out["y_hat"],
        }

    def decompress(self, strings: List[List[bytes]], shape: Dict[str, Tuple[int, ...]], **kwargs) -> Dict[str, Any]:
        *y_strings_, z_strings = strings
        # assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec["hyper"].decompress([z_strings], shape["hyper"])
        y_out = self.latent_codec["y"].decompress(y_strings_, hyper_out["params"])
        return {"y_hat": y_out["y_hat"]}


class CustomCheckerboardLatentCodec(LatentCodec):
    """Reconstructs latent using 2-pass context model with checkerboard anchors.

    Checkerboard context model introduced in [He2021].

    See :py:class:`~compressai.models.sensetime.Cheng2020AnchorCheckerboard`
    for example usage.

    - `forward_method="onepass"` is fastest, but does not use
      quantization based on the intermediate means.
      Uses noise to model quantization.
    - `forward_method="twopass"` is slightly slower, but accurately
      quantizes via STE based on the intermediate means.
      Uses the same operations as [Chandelier2023].
    - `forward_method="twopass_faster"` uses slightly fewer
      redundant operations.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    .. warning:: This implementation assumes that ``entropy_parameters``
       is a pointwise function, e.g., a composition of 1x1 convs and
       pointwise nonlinearities.

    .. code-block:: none

        0. Input:

        □ □ □ □
        □ □ □ □
        □ □ □ □

        1. Decode anchors:

        ◌ □ ◌ □
        □ ◌ □ ◌
        ◌ □ ◌ □

        2. Decode non-anchors:

        ■ ◌ ■ ◌
        ◌ ■ ◌ ■
        ■ ◌ ■ ◌

        3. End result:

        ■ ■ ■ ■
        ■ ■ ■ ■
        ■ ■ ■ ■

        LEGEND:
        ■   decoded
        ◌   currently decoding
        □   empty
    """

    latent_codec: Mapping[str, LatentCodec]

    entropy_parameters: nn.Module
    context_prediction: CheckerboardMaskedConv2d

    def __init__(
        self,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        entropy_parameters: Optional[nn.Module] = None,
        context_prediction: Optional[nn.Module] = None,
        anchor_parity="even",
        forward_method="twopass",
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self.anchor_parity = anchor_parity
        self.non_anchor_parity = {"odd": "even", "even": "odd"}[anchor_parity]
        self.forward_method = forward_method
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.context_prediction = context_prediction or nn.Identity()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": lambda: CustomGaussianConditionalLatentCodec(quantizer="ste"),
            },
            save_direct=True,
        )

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        if self.forward_method == "onepass":
            return self._forward_onepass(y, side_params)
        if self.forward_method == "twopass":
            return self._forward_twopass(y, side_params)
        if self.forward_method == "twopass_faster":
            return self._forward_twopass_faster(y, side_params)
        raise ValueError(f"Unknown forward method: {self.forward_method}")

    def _forward_onepass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Fast estimation with single pass of the entropy parameters network.

        It is faster than the twopass method (only one pass required!),
        but also less accurate.

        This method uses uniform noise to roughly model quantization.
        """
        y_hat = self.quantize(y)
        y_ctx = self._keep_only(self.context_prediction(y_hat), "non_anchor")
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        The first pass gets ``y_hat`` and ``means_hat`` for the anchors.
        This ``y_hat`` is used as context to predict the non-anchors.
        The second pass gets ``y_hat`` for the non-anchors.
        The two ``y_hat`` tensors are then combined. The resulting
        ``y_hat`` models the effects of quantization more realistically.

        To compute ``y_hat_anchors``, we need the predicted ``means_hat``:
        ``y_hat = quantize_ste(y - means_hat) + means_hat``.
        Thus, two passes of ``entropy_parameters`` are necessary.

        """
        B, C, H, W = y.shape

        params = y.new_zeros((B, C * 2, H, W))

        y_hat_anchors = self._forward_twopass_step(y, side_params, params, self._y_ctx_zero(y), "anchor")

        y_hat_non_anchors = self._forward_twopass_step(
            y, side_params, params, self.context_prediction(y_hat_anchors), "non_anchor"
        )

        y_hat = y_hat_anchors + y_hat_non_anchors
        y_out = self.latent_codec["y"](y, params)

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass_step(
        self, y: Tensor, side_params: Tensor, params: Tensor, y_ctx: Tensor, step: str
    ) -> Dict[str, Any]:
        # NOTE: The _i variables contain only the current step's pixels.
        assert step in ("anchor", "non_anchor")

        params_i = self.entropy_parameters(self.merge(y_ctx, side_params))

        # Save params for current step. This is later used for entropy estimation.
        self._copy(params, params_i, step)

        # Apply latent_codec's "entropy_parameters()", if it exists. Usually identity.
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params_i = func(params_i)

        # Keep only elements needed for current step.
        # It's not necessary to mask the rest out just yet, but it doesn't hurt.
        params_i = self._keep_only(params_i, step)
        y_i = self._keep_only(y, step)

        # Determine y_hat for current step, and mask out the other pixels.
        _, means_i = self.latent_codec["y"]._chunk(params_i)
        y_hat_i = self._keep_only(quantize_ste(y_i - means_i) + means_i, step)

        return y_hat_i

    def _forward_twopass_faster(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        This version was written based on the paper description.
        It is a tiny bit faster than the twopass method since
        it avoids a few redundant operations. The "probably unnecessary"
        operations can likely be removed as well.
        The speedup is very small, however.
        """
        y_ctx = self._y_ctx_zero(y)
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params = func(params)
        params = self._keep_only(params, "anchor")  # Probably unnecessary.
        _, means_hat = self.latent_codec["y"]._chunk(params)
        y_hat_anchors = quantize_ste(y - means_hat) + means_hat
        y_hat_anchors = self._keep_only(y_hat_anchors, "anchor")

        y_ctx = self.context_prediction(y_hat_anchors)
        y_ctx = self._keep_only(y_ctx, "non_anchor")  # Probably unnecessary.
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)

        # Reuse quantized y_hat that was used for non-anchor context prediction.
        y_hat = y_out["y_hat"]
        self._copy(y_hat, y_hat_anchors, "anchor")  # Probably unnecessary.

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    @torch.no_grad()
    def _y_ctx_zero(self, y: Tensor) -> Tensor:
        """Create a zero tensor with correct shape for y_ctx."""
        y_ctx_meta = self.context_prediction(y.to("meta"))
        return y.new_zeros(y_ctx_meta.shape)

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        y_ = self.unembed(y)
        y_strings_ = [None] * 2

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].compress(y_[i], params_i)
            y_hat_[i] = y_out["y_hat"]
            [y_strings_[i]] = y_out["strings"]

        y_hat = self.embed(y_hat_)

        return {
            "strings": y_strings_,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        y_strings_ = strings
        n = side_params.shape[0]  # * modified
        assert len(y_strings_) == 2
        # assert all(len(x) == n for x in y_strings_) # * modified

        c, h, w = shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].decompress([y_strings_[i]], params_i)  # * modified
            y_hat_[i] = y_out["y_hat"]

        y_hat = self.embed(y_hat_)

        return {
            "y_hat": y_hat,
        }

    def unembed(self, y: Tensor) -> Tensor:
        """Separate single tensor into two even/odd checkerboard chunks.

        .. code-block:: none

            ■ □ ■ □         ■ ■   □ □
            □ ■ □ ■   --->  ■ ■   □ □
            ■ □ ■ □         ■ ■   □ □
        """
        n, c, h, w = y.shape
        y_ = y.new_zeros((2, n, c, h, w // 2))
        if self.anchor_parity == "even":
            y_[0, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 1::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 0::2]
        else:
            y_[0, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 0::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 1::2]
        return y_

    def embed(self, y_: Tensor) -> Tensor:
        """Combine two even/odd checkerboard chunks into single tensor.

        .. code-block:: none

            ■ ■   □ □         ■ □ ■ □
            ■ ■   □ □   --->  □ ■ □ ■
            ■ ■   □ □         ■ □ ■ □
        """
        num_chunks, n, c, h, w_half = y_.shape
        assert num_chunks == 2
        y = y_.new_zeros((n, c, h, w_half * 2))
        if self.anchor_parity == "even":
            y[..., 0::2, 0::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 1::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[1, ..., 1::2, :]
        else:
            y[..., 0::2, 1::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 0::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[1, ..., 1::2, :]
        return y

    def _copy(self, dest: Tensor, src: Tensor, step: str) -> None:
        """Copy pixels in the current step."""
        assert step in ("anchor", "non_anchor")
        parity = self.anchor_parity if step == "anchor" else self.non_anchor_parity
        if parity == "even":
            dest[..., 0::2, 0::2] = src[..., 0::2, 0::2]
            dest[..., 1::2, 1::2] = src[..., 1::2, 1::2]
        else:
            dest[..., 0::2, 1::2] = src[..., 0::2, 1::2]
            dest[..., 1::2, 0::2] = src[..., 1::2, 0::2]

    def _keep_only(self, y: Tensor, step: str, inplace: bool = False) -> Tensor:
        """Keep only pixels in the current step, and zero out the rest."""
        return self._mask(
            y,
            parity=self.non_anchor_parity if step == "anchor" else self.anchor_parity,
            inplace=inplace,
        )

    def _mask(self, y: Tensor, parity: str, inplace: bool = False) -> Tensor:
        if not inplace:
            y = y.clone()
        if parity == "even":
            y[..., 0::2, 0::2] = 0
            y[..., 1::2, 1::2] = 0
        elif parity == "odd":
            y[..., 0::2, 1::2] = 0
            y[..., 1::2, 0::2] = 0
        elif parity == "all":
            y[:] = 0
        return y

    def merge(self, *args):
        return torch.cat(args, dim=1)

    def quantize(self, y: Tensor) -> Tensor:
        mode = "noise" if self.training else "dequantize"
        y_hat = EntropyModel.quantize(None, y, mode)
        return y_hat


class SlimCheckerboardLatentCodec(LatentCodec):
    """Reconstructs latent using 2-pass context model with checkerboard anchors.

    Checkerboard context model introduced in [He2021].

    See :py:class:`~compressai.models.sensetime.Cheng2020AnchorCheckerboard`
    for example usage.

    - `forward_method="onepass"` is fastest, but does not use
      quantization based on the intermediate means.
      Uses noise to model quantization.
    - `forward_method="twopass"` is slightly slower, but accurately
      quantizes via STE based on the intermediate means.
      Uses the same operations as [Chandelier2023].
    - `forward_method="twopass_faster"` uses slightly fewer
      redundant operations.

    [He2021]: `"Checkerboard Context Model for Efficient Learned Image
    Compression" <https://arxiv.org/abs/2103.15306>`_, by Dailan He,
    Yaoyan Zheng, Baocheng Sun, Yan Wang, and Hongwei Qin, CVPR 2021.

    [Chandelier2023]: `"ELiC-ReImplemetation"
    <https://github.com/VincentChandelier/ELiC-ReImplemetation>`_, by
    Vincent Chandelier, 2023.

    .. warning:: This implementation assumes that ``entropy_parameters``
       is a pointwise function, e.g., a composition of 1x1 convs and
       pointwise nonlinearities.

    .. code-block:: none

        0. Input:

        □ □ □ □
        □ □ □ □
        □ □ □ □

        1. Decode anchors:

        ◌ □ ◌ □
        □ ◌ □ ◌
        ◌ □ ◌ □

        2. Decode non-anchors:

        ■ ◌ ■ ◌
        ◌ ■ ◌ ■
        ■ ◌ ■ ◌

        3. End result:

        ■ ■ ■ ■
        ■ ■ ■ ■
        ■ ■ ■ ■

        LEGEND:
        ■   decoded
        ◌   currently decoding
        □   empty
    """

    latent_codec: Mapping[str, LatentCodec]

    entropy_parameters: nn.Module
    context_prediction: CheckerboardMaskedConv2d

    def __init__(
        self,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        entropy_parameters: Optional[nn.Module] = None,
        context_prediction: Optional[nn.Module] = None,
        anchor_parity="even",
        forward_method="twopass",
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self.anchor_parity = anchor_parity
        self.non_anchor_parity = {"odd": "even", "even": "odd"}[anchor_parity]
        self.forward_method = forward_method
        self.entropy_parameters = entropy_parameters or nn.Identity()
        self.context_prediction = context_prediction or nn.Identity()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": lambda: CustomGaussianConditionalLatentCodec(quantizer="ste"),
            },
            save_direct=True,
        )

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor, ctx_params: Tensor, side_params: Tensor) -> Dict[str, Any]:
        if self.forward_method == "onepass":
            return self._forward_onepass(y, ctx_params, side_params)
        if self.forward_method == "twopass":
            return self._forward_twopass(y, ctx_params, side_params)  # modified # TODO modify compress decompress
        if self.forward_method == "twopass_faster":
            return self._forward_twopass_faster(y, ctx_params, side_params)
        raise ValueError(f"Unknown forward method: {self.forward_method}")

    def _forward_onepass(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Fast estimation with single pass of the entropy parameters network.

        It is faster than the twopass method (only one pass required!),
        but also less accurate.

        This method uses uniform noise to roughly model quantization.
        """
        y_hat = self.quantize(y)
        y_ctx = self._keep_only(self.context_prediction(y_hat), "non_anchor")
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass(self, y: Tensor, ctx_params: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        The first pass gets ``y_hat`` and ``means_hat`` for the anchors.
        This ``y_hat`` is used as context to predict the non-anchors.
        The second pass gets ``y_hat`` for the non-anchors.
        The two ``y_hat`` tensors are then combined. The resulting
        ``y_hat`` models the effects of quantization more realistically.

        To compute ``y_hat_anchors``, we need the predicted ``means_hat``:
        ``y_hat = quantize_ste(y - means_hat) + means_hat``.
        Thus, two passes of ``entropy_parameters`` are necessary.

        """
        B, C, H, W = y.shape

        params = y.new_zeros((B, C * 2, H, W))

        y_hat_anchors = self._forward_twopass_step(y, side_params, params, self._y_ctx_zero(y), "anchor")

        y_hat_non_anchors = self._forward_twopass_step(
            y, side_params, params, self.context_prediction(y_hat_anchors), "non_anchor"
        )

        y_hat = y_hat_anchors + y_hat_non_anchors
        y_out = self.latent_codec["y"](y, params)

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    def _forward_twopass_step(
        self, y: Tensor, side_params: Tensor, params: Tensor, y_ctx: Tensor, step: str
    ) -> Dict[str, Any]:
        # NOTE: The _i variables contain only the current step's pixels.
        assert step in ("anchor", "non_anchor")

        params_i = self.entropy_parameters(self.merge(y_ctx, side_params))

        # Save params for current step. This is later used for entropy estimation.
        self._copy(params, params_i, step)

        # Apply latent_codec's "entropy_parameters()", if it exists. Usually identity.
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params_i = func(params_i)

        # Keep only elements needed for current step.
        # It's not necessary to mask the rest out just yet, but it doesn't hurt.
        params_i = self._keep_only(params_i, step)
        y_i = self._keep_only(y, step)

        # Determine y_hat for current step, and mask out the other pixels.
        _, means_i = self.latent_codec["y"]._chunk(params_i)
        y_hat_i = self._keep_only(quantize_ste(y_i - means_i) + means_i, step)

        return y_hat_i

    def _forward_twopass_faster(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        """Runs the entropy parameters network in two passes.

        This version was written based on the paper description.
        It is a tiny bit faster than the twopass method since
        it avoids a few redundant operations. The "probably unnecessary"
        operations can likely be removed as well.
        The speedup is very small, however.
        """
        y_ctx = self._y_ctx_zero(y)
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        func = getattr(self.latent_codec["y"], "entropy_parameters", lambda x: x)
        params = func(params)
        params = self._keep_only(params, "anchor")  # Probably unnecessary.
        _, means_hat = self.latent_codec["y"]._chunk(params)
        y_hat_anchors = quantize_ste(y - means_hat) + means_hat
        y_hat_anchors = self._keep_only(y_hat_anchors, "anchor")

        y_ctx = self.context_prediction(y_hat_anchors)
        y_ctx = self._keep_only(y_ctx, "non_anchor")  # Probably unnecessary.
        params = self.entropy_parameters(self.merge(y_ctx, side_params))
        y_out = self.latent_codec["y"](y, params)

        # Reuse quantized y_hat that was used for non-anchor context prediction.
        y_hat = y_out["y_hat"]
        self._copy(y_hat, y_hat_anchors, "anchor")  # Probably unnecessary.

        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
            "y_hat": y_hat,
        }

    @torch.no_grad()
    def _y_ctx_zero(self, y: Tensor) -> Tensor:
        """Create a zero tensor with correct shape for y_ctx."""
        y_ctx_meta = self.context_prediction(y.to("meta"))
        return y.new_zeros(y_ctx_meta.shape)

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        n, c, h, w = y.shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)
        y_ = self.unembed(y)
        y_strings_ = [None] * 2

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].compress(y_[i], params_i)
            y_hat_[i] = y_out["y_hat"]
            [y_strings_[i]] = y_out["strings"]

        y_hat = self.embed(y_hat_)

        return {
            "strings": y_strings_,
            "shape": y_hat.shape[1:],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Tuple[int, ...],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        y_strings_ = strings
        n = side_params.shape[0]  # * modified
        assert len(y_strings_) == 2
        # assert all(len(x) == n for x in y_strings_) # * modified

        c, h, w = shape
        y_hat_ = side_params.new_zeros((2, n, c, h, w // 2))
        side_params_ = self.unembed(side_params)

        for i in range(2):
            y_ctx_i = self.unembed(self.context_prediction(self.embed(y_hat_)))[i]
            if i == 0:
                y_ctx_i = self._mask(y_ctx_i, "all")
            params_i = self.entropy_parameters(self.merge(y_ctx_i, side_params_[i]))
            y_out = self.latent_codec["y"].decompress([y_strings_[i]], params_i)  # * modified
            y_hat_[i] = y_out["y_hat"]

        y_hat = self.embed(y_hat_)

        return {
            "y_hat": y_hat,
        }

    def unembed(self, y: Tensor) -> Tensor:
        """Separate single tensor into two even/odd checkerboard chunks.

        .. code-block:: none

            ■ □ ■ □         ■ ■   □ □
            □ ■ □ ■   --->  ■ ■   □ □
            ■ □ ■ □         ■ ■   □ □
        """
        n, c, h, w = y.shape
        y_ = y.new_zeros((2, n, c, h, w // 2))
        if self.anchor_parity == "even":
            y_[0, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 1::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 0::2]
        else:
            y_[0, ..., 0::2, :] = y[..., 0::2, 1::2]
            y_[0, ..., 1::2, :] = y[..., 1::2, 0::2]
            y_[1, ..., 0::2, :] = y[..., 0::2, 0::2]
            y_[1, ..., 1::2, :] = y[..., 1::2, 1::2]
        return y_

    def embed(self, y_: Tensor) -> Tensor:
        """Combine two even/odd checkerboard chunks into single tensor.

        .. code-block:: none

            ■ ■   □ □         ■ □ ■ □
            ■ ■   □ □   --->  □ ■ □ ■
            ■ ■   □ □         ■ □ ■ □
        """
        num_chunks, n, c, h, w_half = y_.shape
        assert num_chunks == 2
        y = y_.new_zeros((n, c, h, w_half * 2))
        if self.anchor_parity == "even":
            y[..., 0::2, 0::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 1::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[1, ..., 1::2, :]
        else:
            y[..., 0::2, 1::2] = y_[0, ..., 0::2, :]
            y[..., 1::2, 0::2] = y_[0, ..., 1::2, :]
            y[..., 0::2, 0::2] = y_[1, ..., 0::2, :]
            y[..., 1::2, 1::2] = y_[1, ..., 1::2, :]
        return y

    def _copy(self, dest: Tensor, src: Tensor, step: str) -> None:
        """Copy pixels in the current step."""
        assert step in ("anchor", "non_anchor")
        parity = self.anchor_parity if step == "anchor" else self.non_anchor_parity
        if parity == "even":
            dest[..., 0::2, 0::2] = src[..., 0::2, 0::2]
            dest[..., 1::2, 1::2] = src[..., 1::2, 1::2]
        else:
            dest[..., 0::2, 1::2] = src[..., 0::2, 1::2]
            dest[..., 1::2, 0::2] = src[..., 1::2, 0::2]

    def _keep_only(self, y: Tensor, step: str, inplace: bool = False) -> Tensor:
        """Keep only pixels in the current step, and zero out the rest."""
        return self._mask(
            y,
            parity=self.non_anchor_parity if step == "anchor" else self.anchor_parity,
            inplace=inplace,
        )

    def _mask(self, y: Tensor, parity: str, inplace: bool = False) -> Tensor:
        if not inplace:
            y = y.clone()
        if parity == "even":
            y[..., 0::2, 0::2] = 0
            y[..., 1::2, 1::2] = 0
        elif parity == "odd":
            y[..., 0::2, 1::2] = 0
            y[..., 1::2, 0::2] = 0
        elif parity == "all":
            y[:] = 0
        return y

    def merge(self, *args):
        return torch.cat(args, dim=1)

    def quantize(self, y: Tensor) -> Tensor:
        mode = "noise" if self.training else "dequantize"
        y_hat = EntropyModel.quantize(None, y, mode)
        return y_hat


class CustomChannelGroupsLatentCodec(LatentCodec):
    """Reconstructs groups of channels using previously decoded groups.

    Context model from [Minnen2020] and [He2022].
    Also known as a "channel-conditional" (CC) entropy model.

    See :py:class:`~compressai.models.sensetime.Elic2022Official`
    for example usage.

    [Minnen2020]: `"Channel-wise Autoregressive Entropy Models for
    Learned Image Compression" <https://arxiv.org/abs/2007.08739>`_, by
    David Minnen, and Saurabh Singh, ICIP 2020.

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.
    """

    latent_codec: Mapping[str, LatentCodec]

    channel_context: Mapping[str, nn.Module]

    def __init__(
        self,
        latent_codec: Optional[Mapping[str, LatentCodec]] = None,
        channel_context: Optional[Mapping[str, nn.Module]] = None,
        *,
        groups: List[int],
        **kwargs,
    ):
        super().__init__()
        self._kwargs = kwargs
        self.groups = list(groups)
        self.groups_acc = list(accumulate(self.groups, initial=0))
        self.channel_context = nn.ModuleDict(channel_context)
        self.latent_codec = nn.ModuleDict(latent_codec)

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        y_ = torch.split(y, self.groups, dim=1)
        y_out_ = [{}] * len(self.groups)
        y_hat_ = [Tensor()] * len(self.groups)
        y_likelihoods_ = [Tensor()] * len(self.groups)

        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f"y{k}"](y_[k], params)
            y_hat_[k] = y_out_[k]["y_hat"]
            y_likelihoods_[k] = y_out_[k]["likelihoods"]["y"]

        y_hat = torch.cat(y_hat_, dim=1)
        y_likelihoods = torch.cat(y_likelihoods_, dim=1)

        return {
            "likelihoods": {
                "y": y_likelihoods,
            },
            "y_hat": y_hat,
        }

    def compress(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        y_ = torch.split(y, self.groups, dim=1)
        y_out_ = [{}] * len(self.groups)
        y_hat = torch.zeros_like(y)
        y_hat_ = y_hat.split(self.groups, dim=1)

        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f"y{k}"].compress(y_[k], params)
            y_hat_[k][:] = y_out_[k]["y_hat"]

        y_strings_groups = [y_out["strings"] for y_out in y_out_]
        assert all(len(y_strings_groups[0]) == len(ss) for ss in y_strings_groups)

        return {
            "strings": [s for ss in y_strings_groups for s in ss],
            "shape": [y_out["shape"] for y_out in y_out_],
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: List[Tuple[int, ...]],
        side_params: Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        n = side_params.shape[0]  # * modified
        # assert all(len(ss) == n for ss in strings) # * modified
        strings_per_group = len(strings) // len(self.groups)

        y_out_ = [{}] * len(self.groups)
        y_shape = (sum(s[0] for s in shape), *shape[0][1:])
        y_hat = torch.zeros((n, *y_shape), device=side_params.device)
        y_hat_ = y_hat.split(self.groups, dim=1)

        for k in range(len(self.groups)):
            params = self._get_ctx_params(k, side_params, y_hat_)
            y_out_[k] = self.latent_codec[f"y{k}"].decompress(
                strings[strings_per_group * k : strings_per_group * (k + 1)],
                shape[k],
                params,
            )
            y_hat_[k][:] = y_out_[k]["y_hat"]

        return {
            "y_hat": y_hat,
        }

    def merge_y(self, *args):
        return torch.cat(args, dim=1)

    def merge_params(self, *args):
        return torch.cat(args, dim=1)

    def _get_ctx_params(self, k: int, side_params: Tensor, y_hat_: List[Tensor]) -> Tensor:
        if k == 0:
            return side_params
        ch_ctx_params = self.channel_context[f"y{k}"](self.merge_y(*y_hat_[:k]))
        return self.merge_params(ch_ctx_params, side_params)


class CondHyperpriorLatentCodec(LatentCodec):

    latent_codec: Mapping[str, LatentCodec]

    def __init__(self, latent_codec: Optional[Mapping[str, LatentCodec]] = None, **kwargs):
        super().__init__()
        self._set_group_defaults(
            "latent_codec",
            latent_codec,
            defaults={
                "y": CustomGaussianConditionalLatentCodec,
                "hyper": CondHyperLatentCodec,
            },
            save_direct=True,
        )

    def __getitem__(self, key: str) -> LatentCodec:
        return self.latent_codec[key]

    def forward(self, y: Tensor, cond: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"](y, cond)
        y_out = self.latent_codec["y"](y, hyper_out["params"])
        return {
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
            "y_hat": y_out["y_hat"],
        }

    def compress(self, y: Tensor, cond: Tensor) -> Dict[str, Any]:
        hyper_out = self.latent_codec["hyper"].compress(y, cond)
        y_out = self.latent_codec["y"].compress(y, hyper_out["params"])
        [z_strings] = hyper_out["strings"]
        return {
            "strings": [*y_out["strings"], z_strings],
            "shape": {"y": y_out["shape"], "hyper": hyper_out["shape"]},
            "y_hat": y_out["y_hat"],
            "cond": cond,
        }

    def decompress(
        self, strings: List[List[bytes]], shape: Dict[str, Tuple[int, ...]], cond: Tensor, **kwargs
    ) -> Dict[str, Any]:
        *y_strings_, z_strings = strings
        # assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        hyper_out = self.latent_codec["hyper"].decompress([z_strings], shape["hyper"], cond)
        y_out = self.latent_codec["y"].decompress(y_strings_, shape["y"], hyper_out["params"])
        return {"y_hat": y_out["y_hat"]}
