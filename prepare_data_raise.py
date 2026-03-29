import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse
import rawpy
import imageio
from multiprocessing import Pool, cpu_count
import lmdb
import pickle
from utils.preprocess import bayer_to_rggb, fix_raw
import exifread

patch_sz = 128
jpg_half = True
to_rggb = True


def process_render(args):
    raw_path, jpg_path = args
    try:
        with rawpy.imread(raw_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_color=rawpy.ColorSpace.sRGB,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                bright=1.0,
                highlight_mode=rawpy.HighlightMode.Clip,
                half_size=jpg_half,
                user_flip=0,
            )
        imageio.imwrite(jpg_path, rgb)
    except Exception as e:
        print(f"Failed to process {os.path.basename(raw_path)}: {e}")


def render_raw_to_rgb(raw_files, render_dir):
    os.makedirs(render_dir, exist_ok=True)
    tasks = [(os.path.join(raw_dir, f), os.path.join(render_dir, os.path.splitext(f)[0] + ".jpg")) for f in raw_files]
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap_unordered(process_render, tasks), total=len(tasks), desc="Rendering RAW to RGB"))


def patchify_image(img, patch_size):
    H, W = img.shape[:2]
    patches = []
    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patch = img[i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches


def process_patches(args):
    raw_path, render_path, raw_patch_dir, render_patch_dir = args
    fname = os.path.basename(raw_path)
    try:
        raw = rawpy.imread(raw_path)
        raw_img = raw.raw_image_visible
    except Exception as e:
        print(f"Error reading RAW {fname}: {e}")
        return
    render_img = np.array(Image.open(render_path).convert("RGB"))
    raw_patches = patchify_image(raw_img, patch_sz)
    render_patches = patchify_image(render_img, patch_sz)
    for idx, (r_patch, rd_patch) in enumerate(zip(raw_patches, render_patches)):
        raw_patch_path = os.path.join(raw_patch_dir, f"{os.path.splitext(fname)[0]}_{idx:05d}.npy")
        render_patch_path = os.path.join(render_patch_dir, f"{os.path.splitext(fname)[0]}_{idx:05d}.png")
        np.save(raw_patch_path, r_patch)
        imageio.imwrite(render_patch_path, rd_patch)


def extract_patches(raw_dir, raw_patch_dir, render_dir, render_patch_dir):
    os.makedirs(raw_patch_dir, exist_ok=True)
    os.makedirs(render_patch_dir, exist_ok=True)
    raw_files = sorted([f for f in os.listdir(raw_dir)])
    tasks = []
    for fname in raw_files:
        raw_path = os.path.join(raw_dir, fname)
        render_path = os.path.join(render_dir, os.path.splitext(fname)[0] + ".jpg")
        tasks.append((raw_path, render_path, raw_patch_dir, render_patch_dir))
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap_unordered(process_patches, tasks), total=len(tasks), desc="Extracting Patches"))


def extract_patches_lmdb(raw_files, lmdb_raw_path, render_dir, lmdb_render_path, raw_dir, split):
    raw_env = lmdb.open(lmdb_raw_path, map_size=50 * 1024 * 1024 * 1024)
    render_env = lmdb.open(lmdb_render_path, map_size=50 * 1024 * 1024 * 1024)

    with raw_env.begin(write=True) as raw_txn, render_env.begin(write=True) as render_txn:

        for fname in tqdm(raw_files, desc="Extracting Patches to LMDB"):
            raw_path = os.path.join(raw_dir, fname)

            render_path = os.path.join(render_dir, os.path.splitext(fname)[0] + ".jpg")

            fname = os.path.basename(raw_path)
            try:
                raw_img = rawpy.imread(raw_path)
                raw_img = raw_img.raw_image_visible if "NikonD40" not in fname else raw_img.raw_image
                orientation = str(exifread.process_file(open(raw_path, "rb")).get("Image Orientation"))
            except Exception as e:
                print(f"Error reading RAW {fname}: {e}")
                return

            render_img = np.array(Image.open(render_path).convert("RGB"))
            raw_img = bayer_to_rggb(raw_img) if to_rggb else raw_img
            keys = []
            # rotate if needed
            # if orientation == "Rotated 90 CW":
            #     raw_img = np.rot90(raw_img, k=3)  # Rotate 270 degrees
            #     render_img = np.rot90(render_img, k=3)
            # elif orientation == "Rotated 90 CCW":
            #     raw_img = np.rot90(raw_img, k=1)  # Rotate 90 degrees
            #     render_img = np.rot90(render_img, k=1)
            # elif orientation == "Rotated 180":
            #     raw_img = np.rot90(raw_img, k=2)  # Rotate 180 degrees
            #     render_img = np.rot90(render_img, k=2)

            assert raw_img.shape[:2] == render_img.shape[:2], (raw_img.shape, render_img.shape, fname)
            raw_patches = patchify_image(raw_img, patch_sz)
            render_patches = patchify_image(render_img, patch_sz)
            for idx, (r_patch, rd_patch) in enumerate(zip(raw_patches, render_patches)):
                key = f"{os.path.splitext(fname)[0]}_{idx:05d}"
                raw_txn.put(key.encode(), pickle.dumps(r_patch))
                render_txn.put(key.encode(), pickle.dumps(rd_patch))
                keys.append(key)

            # Optionally, store the list of keys for this split
            with open(os.path.join(os.path.dirname(lmdb_raw_path), f"{split}_{patch_sz}_keys.txt"), "a") as f:
                for key in keys:
                    f.write(f"{key}\n")


def split_dataset(file_list, train_ratio=0.8, val_ratio=0.1):
    np.random.shuffle(file_list)
    total = len(file_list)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]
    return train_files, val_files, test_files


if __name__ == "__main__":
    root = "/NEW_EDS/JJ_Group/zhengch2506/datasets/nus8"
    data_name = [
        "Canon1DsMkIII",
        "Canon600D",
        "NikonD40",
        "NikonD5200",
        "OlympusEPL6",
        "PanasonicGX1",
        "SamsungNX2000",
        "SonyA57",
    ]
    data_dirs = [os.path.join(root, name) for name in data_name]
    data_dirs = ["/NEW_EDS/JJ_Group/zhengch2506/datasets/raise"]  # for raise dataset
    for data_dir in data_dirs:
        print(f"Processing dataset: {data_dir}")
        out_dir = os.path.join("./data", os.path.basename(data_dir))
        raw_dir = data_dir
        render_dir = os.path.join(out_dir, "RGB") if not jpg_half else os.path.join(out_dir, "RGB_half")
        # raw_patch_dir = (
        #     os.path.join(out_dir, "RAW_P{}".format(patch_sz))
        #     if not bayer_to_rggb
        #     else os.path.join(out_dir, "RAW_RGGB_P{}".format(patch_sz))
        # )
        # render_patch_dir = (
        #     os.path.join(out_dir, "RGB_P{}".format(patch_sz))
        #     if not jpg_half
        #     else os.path.join(out_dir, "RGB_half_P{}".format(patch_sz))
        # )
        raw_patch_lmdb_train = (
            os.path.join(out_dir, "RAW_P{}_train.lmdb".format(patch_sz))
            if not to_rggb
            else os.path.join(out_dir, "RAW_RGGB_P{}_train.lmdb".format(patch_sz))
        )
        render_patch_lmdb_train = (
            os.path.join(out_dir, "RGB_P{}_train.lmdb".format(patch_sz))
            if not jpg_half
            else os.path.join(out_dir, "RGB_half_P{}_train.lmdb".format(patch_sz))
        )

        raw_patch_lmdb_val = (
            os.path.join(out_dir, "RAW_P{}_val.lmdb".format(patch_sz))
            if not to_rggb
            else os.path.join(out_dir, "RAW_RGGB_P{}_val.lmdb".format(patch_sz))
        )
        render_patch_lmdb_val = (
            os.path.join(out_dir, "RGB_P{}_val.lmdb".format(patch_sz))
            if not jpg_half
            else os.path.join(out_dir, "RGB_half_P{}_val.lmdb".format(patch_sz))
        )

        valid_ratio = 1
        valid_ratio = 0.05

        raw_files = np.random.permutation(os.listdir(raw_dir)).tolist()
        raw_files = raw_files[: int(valid_ratio * len(raw_files))]

        train_files, val_files, test_files = split_dataset(raw_files)
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train.txt"), "w") as f:
            for item in train_files:
                f.write("%s\n" % item)
        with open(os.path.join(out_dir, "val.txt"), "w") as f:
            for item in val_files:
                f.write("%s\n" % item)
        with open(os.path.join(out_dir, "test.txt"), "w") as f:
            for item in test_files:
                f.write("%s\n" % item)
        render_raw_to_rgb(raw_files, render_dir)
        # extract_patches(raw_dir, raw_patch_dir, render_dir, render_patch_dir)
        # extract_patches_lmdb(raw_files, raw_patch_lmdb, render_dir, render_patch_lmdb, raw_dir)
        extract_patches_lmdb(train_files, raw_patch_lmdb_train, render_dir, render_patch_lmdb_train, raw_dir, "train")
        extract_patches_lmdb(val_files, raw_patch_lmdb_val, render_dir, render_patch_lmdb_val, raw_dir, "val")
