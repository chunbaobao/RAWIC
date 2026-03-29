import os
from torch.utils.data import Dataset
import PIL.Image as Image
import lmdb
import pickle
import random


class RawJpegLMDBDataset(Dataset):
    def __init__(self, root, split="train", patch_sz=128, transform=None):  # decopule patch size?
        self.root = root
        self.transform = transform
        self.patch_sz = patch_sz
        split_file = os.path.join(root, f"{split}.txt")
        with open(split_file, "r") as f:
            valid_file_names = [line.strip() for line in f.readlines()]

        self.lmdb_raw_path = os.path.join(root, f"RAW_RGGB_P{patch_sz}.lmdb")
        self.lmdb_render_path = os.path.join(root, f"RGB_half_P{patch_sz}.lmdb")
        self.raw_env = lmdb.open(self.lmdb_raw_path, readonly=True, lock=False)
        self.render_env = lmdb.open(self.lmdb_render_path, readonly=True, lock=False)
        with self.raw_env.begin() as raw_txn:
            cursor = raw_txn.cursor()
            valid_prefixes = set(file.split(".")[0] for file in valid_file_names)
            self.keys = []

            for key, _ in cursor:
                key_str = key.decode()
                parts = key_str.split("_")
                prefix = "_".join(parts[:2]) if len(parts) == 3 else parts[0]
                if prefix in valid_prefixes:
                    self.keys.append(key_str)

    def __getitem__(self, index):
        key = self.keys[index]
        with self.raw_env.begin() as raw_txn, self.render_env.begin() as render_txn:
            raw_patch = pickle.loads(raw_txn.get(key.encode()))
            rgb_patch = pickle.loads(render_txn.get(key.encode()))
        if self.transform is not None:
            raw_patch, rgb_patch = self.transform(raw_patch, rgb_patch)
        return raw_patch, rgb_patch

    def __len__(self):
        return len(self.keys)


class RawJpegMaskLMDBDataset(Dataset):
    def __init__(self, root, split="train", patch_sz=128, jpeg_mask_ratio=0.0, transform=None):
        self.root = root
        self.transform = transform
        self.patch_sz = patch_sz
        self.jpeg_mask_ratio = jpeg_mask_ratio

        keys_file = os.path.join(root, f"{split}_{patch_sz}_keys.txt")
        with open(keys_file, "r") as f:
            self.keys = [line.strip() for line in f.readlines()]

        self.lmdb_raw_path = os.path.join(root, f"RAW_RGGB_P{patch_sz}_{split}.lmdb")
        self.lmdb_render_path = os.path.join(root, f"RGB_half_P{patch_sz}_{split}.lmdb")
        self.raw_env = lmdb.open(self.lmdb_raw_path, readonly=True, lock=False)
        self.render_env = lmdb.open(self.lmdb_render_path, readonly=True, lock=False)

    def __getitem__(self, index):
        key = self.keys[index]
        with self.raw_env.begin() as raw_txn, self.render_env.begin() as render_txn:
            raw_patch = pickle.loads(raw_txn.get(key.encode()))
            rgb_patch = pickle.loads(render_txn.get(key.encode()))
        if self.transform is not None:
            raw_patch, rgb_patch = self.transform(raw_patch, rgb_patch)

            if random.random() < self.jpeg_mask_ratio:
                rgb_patch = rgb_patch * 0.0  # zero mask

        return raw_patch, rgb_patch

    def __len__(self):
        return len(self.keys)


class RawJpegMaskLossyLosslessLMDBDataset(Dataset):
    def __init__(self, root, split="train", patch_sz=128, jpeg_mask_ratio=0.0, transform=None):
        self.root = root
        self.transform = transform
        self.patch_sz = patch_sz
        self.jpeg_mask_ratio = jpeg_mask_ratio

        keys_file = os.path.join(root, f"{split}_{patch_sz}_keys.txt")
        with open(keys_file, "r") as f:
            self.keys = [line.strip() for line in f.readlines()]

        self.lmdb_raw_path = os.path.join(root, f"RAW_RGGB_P{patch_sz}_{split}.lmdb")
        self.lmdb_render_path = os.path.join(root, f"RGB_half_P{patch_sz}_{split}.lmdb")
        self.raw_env = lmdb.open(self.lmdb_raw_path, readonly=True, lock=False)
        self.render_env = lmdb.open(self.lmdb_render_path, readonly=True, lock=False)

    def __getitem__(self, index):
        key = self.keys[index]
        with self.raw_env.begin() as raw_txn, self.render_env.begin() as render_txn:
            raw_patch = pickle.loads(raw_txn.get(key.encode()))
            rgb_patch = pickle.loads(render_txn.get(key.encode()))
        if self.transform is not None:
            raw_patch, rgb_patch = self.transform(raw_patch, rgb_patch)

            if random.random() < self.jpeg_mask_ratio:
                rgb_patch = rgb_patch * 0.0  # zero mask

        if "Nikon" in key:  # nikon is lossy raw image
            return raw_patch, rgb_patch, 1  # lossy
        else:
            return raw_patch, rgb_patch, 0  # lossless

    def __len__(self):
        return len(self.keys)


class ImgDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.imgs = os.listdir(self.root)
        self.imgs.sort()

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0  # dummy rgb

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    patch_sz = 128
    root = "./data/Canon600D"
    root = "./data/nikon"
    dataset = RawJpegLMDBDataset(root, split="train", patch_sz=patch_sz)

    print(f"Dataset length: {len(dataset)}")
    import time
    import tqdm

    start_time = time.time()

    for raw_patch, render_patch in tqdm.tqdm(dataset):
        pass
    end_time = time.time()
    print(f"Time taken to iterate through dataset: {end_time - start_time} seconds")
