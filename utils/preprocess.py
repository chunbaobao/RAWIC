import torch.nn.functional as F
import torch.nn as nn
import torch


def img2patch(img, patch_sz):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    pad_h = (patch_sz - H % patch_sz) % patch_sz
    pad_w = (patch_sz - W % patch_sz) % patch_sz
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
    patches = img_pad.unfold(2, patch_sz, patch_sz).unfold(3, patch_sz, patch_sz)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, C, patch_sz, patch_sz)

    return patches


def patchify_image(img, patch_size):
    H, W = img.shape[:2]
    patches = []
    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patch = img[i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches


def patch2img(patch, img_sz):
    C = patch.shape[1]
    patch_sz = patch.shape[2]
    H, W = img_sz
    pad_h = (patch_sz - H % patch_sz) % patch_sz
    pad_w = (patch_sz - W % patch_sz) % patch_sz
    rows = (H + pad_h) // patch_sz
    cols = (W + pad_w) // patch_sz
    patch = patch.view(-1, rows, cols, C, patch_sz, patch_sz)
    patch = patch.permute(0, 3, 1, 4, 2, 5).contiguous()
    img = patch.view(-1, C, H + pad_h, W + pad_w)
    img = img[:, :, :H, :W]
    return img


class SpaceToDepth(nn.Module):
    def __init__(self, k, rerange=False):
        super(SpaceToDepth, self).__init__()
        self.k = k
        self.rerange = rerange

    def forward(self, x):
        x = nn.PixelUnshuffle(self.k)(x)
        if self.rerange and x.size(1) == 3 * self.k**2:
            index = torch.tensor([[k + self.k**2 * i for i in range(3)] for k in range(self.k**2)]).flatten()
            x = x.index_select(1, index.to(x.device))
        return x


class DepthToSpace(nn.Module):
    def __init__(self, k, rerange=False):
        super(DepthToSpace, self).__init__()
        self.k = k
        self.rerange = rerange

    def forward(self, x):
        if self.rerange:
            index = torch.tensor(
                [[i + 3 * k for k in range(self.k**2)] for i in range(3)]
            ).flatten()  # [0,3,6,...,1,4,7,...,2,5,8,...]
            x = x.index_select(1, index.to(x.device))
        x = nn.PixelShuffle(self.k)(x)
        return x


import numpy as np


def bayer_to_rggb(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:

    H, W = raw.shape

    if H % 2 != 0:
        raw = raw[:-1, :]
        H -= 1
    if W % 2 != 0:
        raw = raw[:, :-1]
        W -= 1

    pattern = pattern.upper()
    if pattern not in ["RGGB", "BGGR", "GRBG", "GBRG"]:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    R = raw[0:H:2, 0:W:2]
    Gr = raw[0:H:2, 1:W:2]
    Gb = raw[1:H:2, 0:W:2]
    B = raw[1:H:2, 1:W:2]

    if pattern == "RGGB":
        pass
    elif pattern == "BGGR":
        R, B = B, R
        Gr, Gb = Gb, Gr
    elif pattern == "GRBG":
        R, Gr, Gb, B = Gr, R, B, Gb
    elif pattern == "GBRG":
        R, Gr, Gb, B = Gb, B, R, Gr

    rggb_raw = np.stack([R, Gr, Gb, B], axis=-1)
    return rggb_raw


def fix_raw(raw_path):
    # 1. Fix rotation based on EXIF

    import rawpy
    import exifread

    raw = rawpy.imread(raw_path)
    raw_img = raw.raw_image_visible
    # raw_img = (raw_img / raw.white_level).astype(np.float32)

    # bit_depth = int(np.ceil(np.log2(raw.white_level + 1)))
    # shift = 16 - bit_depth
    # raw_img = (raw_data.astype(np.uint16) << shift) / 65535.0
    # raw_img = (raw_data) / 16383.0
    # raw_img = raw_img.astype(np.float32)

    # rotate raw image if needed
    with open(raw_path, "rb") as f:
        tags = exifread.process_file(f)

        orientation_tag = tags.get("Image Orientation")

        if orientation_tag:
            orientation = str(orientation_tag)
            if orientation == "Rotated 90 CW":
                raw_img = np.rot90(raw_img, k=3)  # Rotate 270 degrees
            elif orientation == "Rotated 90 CCW":
                raw_img = np.rot90(raw_img, k=1)  # Rotate 90 degrees
            elif orientation == "Rotated 180":
                raw_img = np.rot90(raw_img, k=2)  # Rotate 180 degrees
    return raw_img
