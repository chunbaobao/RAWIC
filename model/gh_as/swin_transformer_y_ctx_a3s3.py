import torch
import torch.nn as nn
from compressai.layers import GDN, conv3x3, subpel_conv3x3, ResidualBlock, conv1x1
from model.custom_layers import SWin_Attention, downsample_conv1x1, subpel_conv1x1
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.latent_codecs import LatentCodec


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


class Analysis(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            AnalysisBlock(in_ch, out_ch),
            AnalysisBlock(out_ch, out_ch),
            AnalysisBlock(out_ch, out_ch),
            conv3x3(out_ch, out_ch, stride=2),
        )

    def forward(self, input):
        out = self.layers(input)
        return out


class Synthesis(nn.Module):
    def __init__(self, in_ch, prior_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch),
            SynthesisBlock(in_ch, in_ch),
        )
        self.conv_prior = subpel_conv3x3(in_ch, prior_ch, r=2)

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

        self.layers = nn.Sequential(
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
        out = self.layers(input)
        # return mu, sigma
        return out
