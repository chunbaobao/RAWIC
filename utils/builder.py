import importlib.util
import sys
from pathlib import Path
import argparse
import types


def load_config(config_path):
    module_name = Path(config_path).stem
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config_module
    spec.loader.exec_module(config_module)
    return config_module


def merge_config_args(config: types.ModuleType, args: argparse.Namespace) -> argparse.Namespace:
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}
    args_dict = vars(args)
    for key, value in config_dict.items():
        if key not in args_dict or args_dict[key] is None:
            args_dict[key] = value
    return argparse.Namespace(**args_dict)


def check_args(args: argparse.Namespace):
    if not Path(args.train_path).exists():
        raise FileNotFoundError(f"Training path {args.train_path} does not exist.")
    if not Path(args.val_path).exists():
        raise FileNotFoundError(f"Validation path {args.val_path} does not exist.")
    if args.device not in ["cuda", "cpu"]:
        raise ValueError("Device must be either 'cuda' or 'cpu'.")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if args.num_workers < 0:
        raise ValueError("Number of workers must be a non-negative integer.")
    if hasattr(args, "lr") and args.lr is not None and args.lr <= 0:
        raise ValueError("Learning rate must be a positive float.")
    if hasattr(args, "blr") and args.blr is not None and args.blr <= 0:
        raise ValueError("Base learning rate must be a positive float.")
    if hasattr(args, "aux_lr") and args.aux_lr is not None and args.aux_lr <= 0:
        raise ValueError("Auxiliary learning rate must be a positive float.")
    if hasattr(args, "aux_blr") and args.aux_blr is not None and args.aux_blr <= 0:
        raise ValueError("Auxiliary base learning rate must be a positive float.")
    if args.num_epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")
    if args.lr_reduce_patience < 0:
        raise ValueError("LR reduce patience must be a non-negative integer.")
    if not (0 < args.lr_reduce_factor < 1):
        raise ValueError("LR reduce factor must be between 0 and 1.")
    if not (0 <= args.p_hflip <= 1):
        raise ValueError("Probability of horizontal flip must be between 0 and 1.")
    if not (0 <= args.p_vflip <= 1):
        raise ValueError("Probability of vertical flip must be between 0 and 1.")
    if args.prefetch_factor <= 0:
        raise ValueError("Prefetch factor must be a positive integer.")
    if args.clip_grad is not None and args.clip_grad <= 0:
        raise ValueError("Gradient clipping value must be a positive float or None.")
    # if args.patch_sz % args.block_sz != 0:
    #     raise ValueError("Patch size must be divisible by block size.")

    # args.k = args.patch_sz // args.block_sz  # upscaling factor
    return args


def ckpt2config(ckpt_path: str) -> str:
    import os
    import glob
    import re
    from tensorboard.backend.event_processing import event_accumulator

    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    logs_dir = os.path.join(run_dir, "logs")
    event_file = glob.glob(os.path.join(logs_dir, "events.out.tfevents.*"))[0]
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    event = ea.Tensors("args/text_summary")[0]
    raw_text = event.tensor_proto.string_val[0].decode("utf-8")

    match = re.search(r"config='([^']+)'", raw_text)

    config_path = match.group(1)

    return config_path
