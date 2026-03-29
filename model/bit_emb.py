# bit embedding
import torch
import torch.nn as nn


class BitEmb(nn.Module):
    def __init__(self, num_bits: int, emb_dim: int, out_dim: int):
        super().__init__()
        self.num_bits = num_bits
        self.emb_dim = emb_dim

        self.embeding = nn.Sequential(
            nn.Embedding(num_bits, emb_dim), nn.Linear(emb_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, bit_depth: torch.Tensor):
        # bit_depth: (B,)
        # normalize to [0, num_bits - 1]
        bit_depth = bit_depth.long()
        emb = self.embeding(bit_depth)[:, :, None, None]  # (B, out_dim, 1, 1)
        return emb


class BitEmbPW(nn.Module):
    def __init__(self, num_bits: int, emb_dim: int, out_dim: int):
        super().__init__()
        self.num_bits = num_bits
        self.emb_dim = emb_dim

        self.embeding = nn.Sequential(
            nn.Embedding(num_bits, emb_dim), nn.Linear(emb_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, bit_depth: torch.Tensor):
        # bit_depth: (B,C , H , W)
        # normalize to [0, num_bits - 1]
        bit_depth = bit_depth.long()
        emb = self.embeding(bit_depth).squeeze(-1)
        return emb
