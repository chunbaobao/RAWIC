import torch.nn as nn
from compressai.layers import GDN, conv3x3, subpel_conv3x3, ResidualBlock, conv1x1
from model.custom_layers import SWin_Attention, downsample_conv1x1, subpel_conv1x1
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class AnalysisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=2)
        self.gdn = GDN(out_ch)
        self.skip = conv3x3(in_ch, out_ch, stride=2)
        self.rb = ResidualBlock(out_ch, out_ch)

    def forward(self, input):
        out = self.conv1(input)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out = out + self.skip(input)

        out = self.rb(out)
        return out


class AnalysisBlock_wo_ds(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)
        self.gdn = GDN(out_ch)
        self.skip = conv3x3(in_ch, out_ch, stride=1)
        self.rb = ResidualBlock(out_ch, out_ch)

    def forward(self, input):
        out = self.conv1(input)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)
        out = out + self.skip(input)

        out = self.rb(out)
        return out


class SynthesisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb = ResidualBlock(in_ch, out_ch)
        self.conv_up = subpel_conv3x3(out_ch, out_ch, r=2)
        self.igdn = GDN(out_ch, inverse=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.upsample = subpel_conv3x3(out_ch, out_ch, r=2)

    def forward(self, input):
        out1 = self.rb(input)

        out = self.conv_up(out1)
        out = self.igdn(out)
        out = self.conv(out)
        out = self.leaky_relu(out)

        out = out + self.upsample(out1)

        return out


class SynthesisBlock_wo_us(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rb = ResidualBlock(in_ch, out_ch)
        self.conv_up = subpel_conv3x3(out_ch, out_ch, r=1)
        self.igdn = GDN(out_ch, inverse=True)
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.upsample = subpel_conv3x3(out_ch, out_ch, r=1)

    def forward(self, input):
        out1 = self.rb(input)

        out = self.conv_up(out1)
        out = self.igdn(out)
        out = self.conv(out)
        out = self.leaky_relu(out)

        out = out + self.upsample(out1)

        return out


class Analysis(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            AnalysisBlock_wo_ds(in_ch, out_ch),
            AnalysisBlock_wo_ds(out_ch, out_ch),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=8),
            AnalysisBlock(out_ch, out_ch),
            conv3x3(out_ch, out_ch, stride=2),
            SWin_Attention(dim=out_ch, num_heads=8, window_size=4),
        )

    def forward(self, input):
        out = self.layers(input)
        return out


class Synthesis(nn.Module):
    def __init__(self, in_ch, prior_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            SWin_Attention(dim=in_ch, num_heads=8, window_size=4),
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch),
            SWin_Attention(dim=in_ch, num_heads=8, window_size=8),
            SynthesisBlock_wo_us(in_ch, in_ch),
        )
        self.conv_prior = subpel_conv3x3(in_ch, prior_ch, r=1)

    def forward(self, input):
        out = self.layers(input)
        prior = self.conv_prior(out)
        return prior


class HyperAnalysis(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            downsample_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

    def forward(self, input):
        out = self.layers(input)
        return out


class HyperSynthesis(nn.Module):
    def __init__(self, num_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers_mu = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

        self.layers_sigma = nn.Sequential(
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv1x1(num_ch, num_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv1x1(num_ch, num_ch),
        )

    def forward(self, input):
        mu = self.layers_mu(input)
        sigma = self.layers_sigma(input)
        return mu, sigma


class PriorCompressor(CompressionModel):
    def __init__(self, z_dist: EntropyBottleneck, y_dist: GaussianConditional, ga, gs, ha, hs):
        super().__init__()
        self.z_dist = z_dist
        self.y_dist = y_dist
        self.g_a = ga
        self.g_s = gs
        self.h_a = ha
        self.h_s = hs

    def forward(self, x):
        y = self.g_a(x)  # yscale : 1
        z = self.h_a(y)

        z_hat, z_likelihoods = self.z_dist(z)
        mu_hat, sigma_hat = self.h_s(z_hat)  # TODO add non-gaussian dist
        y_hat, y_likelihoods = self.y_dist(y, sigma_hat, mu_hat)
        prior = self.g_s(y_hat)  # 256 *64 * 64 ->

        return {"prior": prior, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)

        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:], z.size()[:2])

        mu_hat, sigma_hat = self.h_s(z_hat)

        B, C, H, W = mu_hat.shape
        indexes = self.y_dist.build_indexes(sigma_hat.reshape(1, B * C, H, W))
        y_strings = self.y_dist.compress(y.reshape(1, B * C, H, W), indexes, means=mu_hat.reshape(1, B * C, H, W))

        return {"strings": [y_strings, z_strings], "shape": z.size()}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape[2:], shape[:2])
        mu_hat, sigma_hat = self.h_s(z_hat)

        B, C, H, W = mu_hat.shape
        indexes = self.y_dist.build_indexes(sigma_hat.reshape(1, B * C, H, W))
        y_hat = self.y_dist.decompress(strings[0], indexes, means=mu_hat.reshape(1, B * C, H, W))
        y_hat = y_hat.reshape(B, C, H, W)
        return {"prior": self.g_s(y_hat)}
