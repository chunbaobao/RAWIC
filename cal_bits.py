from configs.default import *

from datasets.dataset import RawJpegMaskLMDBDataset
from torch.utils.data import ConcatDataset, DataLoader
import tqdm
from collections import Counter
import os
import torch

# dataset
data_name = [
    "Canon1DsMkIII",
    "Canon600D",
    # "NikonD40",
    # "NikonD5200",
    "OlympusEPL6",
    "PanasonicGX1",
    "SamsungNX2000",
    # "SonyA57",  # for weird 15 bit depth
    "raise",
]
data_paths = [os.path.join("./data", name) for name in data_name]

jpeg_mask_ratio = 0.0

patch_szs = [1, 2, 4, 8, 16, 32, 64]
for patch_sz in patch_szs:
    # transform_val = RAWEvalTransform(patch_sz)

    # val_dataset = ConcatDataset(
    #     [
    #         RawJpegMaskLMDBDataset(data_path, split="train", transform=transform_val, jpeg_mask_ratio=jpeg_mask_ratio)
    #         for data_path in data_paths
    #     ]
    # )

    from datasets.transform import TrainTransform, EvalTransform
    from datasets.dataset import ImgDataset

    val_path = "/NEW_EDS/JJ_Group/zhengch2506/nasic/data/DIV2K_valid_p128"
    transform_val = EvalTransform(patch_sz)
    val_dataset = ImgDataset(val_path, transform=transform_val)

    dataloader = DataLoader(val_dataset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False)

    bit_counter = Counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            raw, jpeg = batch
            raw = raw.to(device, non_blocking=True, dtype=torch.int32)

            bit_depths = torch.ceil(
                torch.log2(torch.max(raw.view(raw.size(0), raw.size(1), -1), dim=-1).values.clip(min=0) + 1)
            )
            bit_depths = bit_depths.cpu().numpy()

            bit_counter.update(bit_depths.flatten().astype(int))
    print(f"Patch size: {patch_sz}")
    print("Bit-depth count =", bit_counter)
