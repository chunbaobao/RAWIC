import imagecodecs
import numpy as np
import os
import subprocess
import tempfile
from PIL import Image
import time
import hashlib
import argparse
import json
import multiprocessing as mp
from tqdm import tqdm
import rawpy
from utils.preprocess import bayer_to_rggb

BAYER_PATTERN = "RGGB"
TMP_DIR = "./tmp"
EXTERNAL_CODEC = ["BPG", "FLIF"]
NOSPPORT16_CODEC = ["WebP", "QOI"]
flif_path = "/home/JJ_Group/zhengch2506/FLIF/src/flif"


def get_md5(*args):
    return hashlib.md5("".join(args).encode()).hexdigest()


def calculate_bpsp(encoded_size, num_pixels):
    return (encoded_size * 8) / num_pixels


def remove_icc(img):
    if "icc_profile" in img.info:
        data = list(img.getdata())
        image_without_icc = Image.new(img.mode, img.size)
        image_without_icc.putdata(data)
        return image_without_icc
    return img


def bpg_encode(img, is_delete=False):
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR) as f:
        # img = remove_icc(img)
        img = Image.fromarray(img)
        img.save(f.name)
        out_name = f.name.replace(".png", ".bpg")
        command = f"bpgenc -lossless {f.name} -o {out_name} -m 9"
        start_time = time.time()
        subprocess.run(command, shell=True)
        enc_time = time.time() - start_time
        stream = open(out_name, "rb").read()
        if is_delete:
            os.remove(out_name)
            return stream
    return (stream, out_name), enc_time


def flif_encode(img, is_delete=False):
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR) as f:
        # img = remove_icc(img)
        img = Image.fromarray(img)
        img.save(f.name)
        out_name = f.name.replace(".png", ".flif")
        command = f"{flif_path} {f.name} {out_name} -N -E40"
        start_time = time.time()
        subprocess.run(command, shell=True)
        enc_time = time.time() - start_time
        stream = open(out_name, "rb").read()
        if is_delete:
            os.remove(out_name)
            return stream
    return (stream, out_name), enc_time


def bpg_decode(path):
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR) as f:
        command = f"bpgdec {path} -o {f.name}"
        start_time = time.time()
        subprocess.run(command, shell=True)
        dec_time = time.time() - start_time
        os.remove(path)
    return dec_time


def flif_decode(path):
    with tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DIR) as f:
        command = f"{flif_path} {path} -o {f.name}"
        start_time = time.time()
        subprocess.run(command, shell=True)
        dec_time = time.time() - start_time
        os.remove(path)
    return dec_time


def calculate_img_bpsp(img_path, codec, codec_name, cache_dir, bayer="C"):
    # img = Image.open(img_path).convert("RGB")
    img = rawpy.imread(img_path)
    img = img.raw_image_visible if "NikonD40" not in img_path else img.raw_image

    img = bayer_to_rggb(img, pattern=bayer) if codec_name not in EXTERNAL_CODEC else img

    cache_file = os.path.join(cache_dir, get_md5(img_path, codec_name, bayer) + ".json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            result = json.load(f)
            return result["bpsp"], result["enc_time"], result["dec_time"]

    enc, dec = codec
    if codec_name in EXTERNAL_CODEC:
        img_encoded, enc_time = enc(img)
        dec_time = dec(img_encoded[1])
        img_encoded = img_encoded[0]

    elif codec_name in NOSPPORT16_CODEC:
        # split the 16bit image into 2 8bit images
        img_8bit1 = (img >> 8).astype(np.uint8)
        img_8bit2 = (img & 0xFF).astype(np.uint8)

        start_time = time.time()
        img_encoded1 = enc(img_8bit1)
        img_encoded2 = enc(img_8bit2)
        enc_time = time.time() - start_time
        start_time = time.time()
        dec_img1 = dec(img_encoded1).astype(np.uint16) << 8
        dec_img2 = dec(img_encoded2).astype(np.uint16)
        dec_img = dec_img1 | dec_img2
        if np.any(dec_img != img):
            print(f"Warning: Decoded image does not match original for {img_path} with {codec_name}")
        dec_time = time.time() - start_time
        img_encoded = [img_encoded1, img_encoded2]

    else:

        start_time = time.time()
        img_encoded = enc(img)
        enc_time = time.time() - start_time

        start_time = time.time()
        dec_img = dec(img_encoded)
        if np.any(dec_img != img):
            print(f"Warning: Decoded image does not match original for {img_path} with {codec_name}")
        dec_time = time.time() - start_time

    if type(img_encoded) is not list:
        img_encoded = [img_encoded]
    bpsp = calculate_bpsp(sum(len(part) for part in img_encoded), img.size)
    result = {"bpsp": bpsp, "enc_time": enc_time, "dec_time": dec_time}

    with open(cache_file, "w") as f:
        json.dump(result, f)

    return bpsp, enc_time, dec_time


def process_dir(imgdir, codec_name, codec, cache_dir, multiproc=False, bayer="C", use_split=False):
    if not use_split:
        img_paths = [os.path.join(imgdir, f) for f in os.listdir(imgdir)]
    else:
        test_txt_path = os.path.join("./data", os.path.basename(imgdir), "test.txt")  # read from split txt file
        with open(test_txt_path, "r") as f:
            img_paths = [os.path.join(imgdir, line.strip()) for line in f.readlines()]

    if multiproc:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        calculate_img_bpsp,
                        [(p, codec, codec_name, cache_dir, bayer) for p in img_paths],
                        chunksize=1,
                    ),
                    total=len(img_paths),
                    desc=f"{codec_name} (Multiproc)",
                )
            )
    else:
        results = [
            calculate_img_bpsp(p, codec, codec_name, cache_dir, bayer=bayer)
            for p in tqdm(img_paths, desc=f"{codec_name} (Single)")
        ]

    bpsps, enc_times, dec_times = zip(*results)

    return {
        "bpsp": np.mean(bpsps),
        "enc_time": np.mean(enc_times),
        "dec_time": np.mean(dec_times),
        "count": len(bpsps),
    }


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgdir",
        type=str,
        nargs="+",
        default=["/NEW_EDS/JJ_Group/zhengch2506/datasets/nus8/Canon1DsMkIII"],
        help="Directory containing images to encode",
    )
    parser.add_argument("--cache_dir", type=str, default=".cache_raw", help="Directory to cache results")
    parser.add_argument("--multiproc", action="store_true", help="Use multiprocessing")
    parser.add_argument("--use_split", action="store_true", help="Use split images for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = config_parser()

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    if isinstance(args.imgdir, str):
        args.imgdir = [args.imgdir]

    ENCODERS = {
        "PNG": (imagecodecs.png_encode, imagecodecs.png_decode),
        "JPEG-LS": (imagecodecs.jpegls_encode, imagecodecs.jpegls_decode),
        "JPEG2000": (imagecodecs.jpeg2k_encode, imagecodecs.jpeg2k_decode),
        "WebP": (imagecodecs.webp_encode, imagecodecs.webp_decode),
        "JPEG-XL": (imagecodecs.jpegxl_encode, imagecodecs.jpegxl_decode),
        "QOI": (imagecodecs.qoi_encode, imagecodecs.qoi_decode),
        # "BPG": (bpg_encode, bpg_decode),
        "FLIF": (flif_encode, flif_decode),
    }

    for codec_name, codec in ENCODERS.items():
        for imgdir in args.imgdir:
            print(f"\n[{codec_name}] Evaluating...")
            print(f"Image Directory: {imgdir}")

            result = process_dir(
                imgdir,
                codec_name,
                codec,
                args.cache_dir,
                multiproc=args.multiproc,
                bayer=BAYER_PATTERN,
                use_split=args.use_split,
            )
            print(
                f"Images: {result['count']}, "
                f"Avg bpsp: {result['bpsp']:.2f}, "
                f"Enc Time: {result['enc_time']:.2f}s, "
                f"Dec Time: {result['dec_time']:.2f}s"
            )
