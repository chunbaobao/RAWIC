import os
import torch
import shutil
import argparse
import numpy as np
import hashlib
import time
from model.distribution.lmm import MixtureLogistic
from model.distribution.rgb_lmm import RGBMixtureLogistic, MultiRGBMixtureLogistic
from model.distribution.rggb_lmm import RGGBMixtureLogistic
from model.distribution.rggb_lmm_bit import RGGBMixtureLogisticDiffBitdepth
from model.distribution.rggb_lmm_bit_pw import RGGBMixtureLogisticDiffBitdepthPixelWise
from model.distribution.rgb_lmm_bit_pw import RGBMixtureLogisticDiffBitdepthPixelWise
from model.distribution.rgb_lmm_bit import RGBMixtureLogisticDiffBitdepth


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_script_dir(path, root_dir=".", exclude_dirs: list = None):
    if not os.path.exists(path):
        os.makedirs(path)

    if isinstance(exclude_dirs, str):
        exclude_dirs = [exclude_dirs]

    scripts_dst = os.path.join(path, "scripts")
    os.makedirs(scripts_dst, exist_ok=True)

    for root, dirs, files in os.walk(root_dir):
        if any(os.path.abspath(root).startswith(os.path.abspath(ex_dir)) for ex_dir in (exclude_dirs)):
            continue  # Skip the experiment directory itself

        # if os.path.abspath(root).startswith(os.path.abspath(exclude_dirs)):
        #     continue  # Skip the experiment directory itself

        rel_dir = os.path.relpath(root, root_dir)
        dst_dir = os.path.join(scripts_dst, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        for f in files:
            if f.endswith(".py"):
                src_file = os.path.join(root, f)
                dst_file = os.path.join(dst_dir, f)
                shutil.copy(src_file, dst_file)


def get_entropy_model_channels(dist):
    if dist == MixtureLogistic:
        return dist.mix_num * 3  # mu, log_sigma, weights for every pixel

    elif (
        dist == RGBMixtureLogistic
        or dist == MultiRGBMixtureLogistic
        or dist == RGBMixtureLogisticDiffBitdepthPixelWise
        or dist == RGBMixtureLogisticDiffBitdepth
    ):
        return dist.mix_num * 10 if dist.no_multichannel_lmm else dist.mix_num * 12

    elif (
        dist == RGGBMixtureLogistic
        or dist == RGGBMixtureLogisticDiffBitdepth
        or dist == RGGBMixtureLogisticDiffBitdepthPixelWise
    ):
        return dist.mix_num * 13 if dist.no_multichannel_lmm else dist.mix_num * 16

    # TODO add more


def filter_args(args: argparse.Namespace):
    filtered_args = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool, type(None))):
            filtered_args[key] = value
    return filtered_args


def get_md5(*paths):
    key = ""
    for path in paths:
        key += path
    return hashlib.md5(key.encode()).hexdigest()


def get_unique_dir(base_dir):

    suffix = 1
    unique_dir = base_dir
    while os.path.exists(unique_dir):
        unique_dir = f"{base_dir}_{suffix}"
        suffix += 1

    return unique_dir


class AverageMeter:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    def __init__(self, results_dict, key):
        self.results = results_dict
        self.key = key

    def __enter__(self):
        self.start = time.time()
        return self  # Optional

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.results[self.key] = time.time() - self.start
