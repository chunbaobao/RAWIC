import torch
import os
from codec_dp_cu_io import compress, decompress
import utils.misc as misc
from utils.preprocess import bayer_to_rggb, img2patch, patch2img
from datasets.transform import raw_to_tensor
import utils.builder as builder
import rawpy
from model.codecwrapper_cu import EncWrapper, DecWrapper
from utils.dist import CustomDP
from tqdm import tqdm
import json
import io
import imageio.v2 as imageio
import numpy as np
from torchvision import transforms
import time

from model.loss import BPPLoss

patch_sz = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument("--imgdir", type=str, nargs="+", help="Directory containing images to encode")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory to cache results")
    parser.add_argument("--use_split", action="store_true", help="Use split images for testing")
    parser.add_argument("--use_jpeg", action="store_true", help="Use JPEG as conditioning")
    parser.add_argument("--dryrun", action="store_true", help="Run dry run for testing purposes")
    return parser.parse_args()


def dryrun(model, raw_path, use_jpeg, criterion=BPPLoss()):
    # read raw image

    raw = rawpy.imread(raw_path)

    # raw preprocess
    raw_img = raw.raw_image_visible if "NikonD40" not in raw_path else raw.raw_image
    hw = raw_img.shape[0] * raw_img.shape[1]
    raw_img = bayer_to_rggb(raw_img)

    if use_jpeg:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_color=rawpy.ColorSpace.sRGB,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            bright=1.0,
            highlight_mode=rawpy.HighlightMode.Clip,
            half_size=True,
            user_flip=0,
        )
        buffer = io.BytesIO()
        imageio.imwrite(buffer, rgb, format="jpeg")
        jpg_bin = buffer.getvalue()
        # jpg_bin = imagecodecs.jpeg_encode(rgb)
        # jpg = imagecodecs.jpeg_decode(jpg_bin)
        jpg = imageio.imread(io.BytesIO(jpg_bin))

    else:
        jpg = np.zeros((raw_img.shape[0], raw_img.shape[1], 3), dtype=np.uint8)
        jpg_bin = b""
    jpeg_bpp = len(jpg_bin) * 8 / hw
    raw_img = raw_to_tensor(raw_img).unsqueeze(0)
    jpg = transforms.ToTensor()(jpg).unsqueeze(0)

    x = img2patch(raw_img, patch_sz=patch_sz).to(device)
    jpg_patch = img2patch(jpg, patch_sz=patch_sz).to(device)

    if torch.cuda.device_count() > 1:
        bit_depth_bpp = (
            x.shape[0]
            * model.module.get_bit_depth_num()
            * torch.log2(torch.tensor(model.module.end_bit - model.module.start_bit + 1))
            / hw
        ).item()
    else:
        bit_depth_bpp = (
            x.shape[0] * model.get_bit_depth_num() * torch.log2(torch.tensor(model.end_bit - model.start_bit + 1)) / hw
        ).item()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        model.eval()
        outs = model(x, jpg_patch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = time.time() - start_time

    if type(outs) is not list:
        outs = [outs]

    # results = sum(criterion(x_chunk, out) for out, x_chunk in zip(outs, x.chunk(torch.cuda.device_count())))
    z_bpp = 0.0
    y_bpp = 0.0
    x_bpp = 0.0
    loss = 0.0
    for out in outs:
        z_bpp += -torch.log2(out["likelihoods"]["z"]).sum().item()
        y_bpp += -torch.log2(out["likelihoods"]["y"]).sum().item()
        x_bpp += -out["likelihoods"]["x"].sum().item() / torch.log(torch.tensor(2.0)).item()

    z_bpp /= hw
    y_bpp /= hw
    x_bpp /= hw
    loss = x_bpp + z_bpp + y_bpp + jpeg_bpp + bit_depth_bpp

    return {
        "loss": loss,
        "x_bpp": x_bpp,
        "y_bpp": y_bpp,
        "z_bpp": z_bpp,
        "jpeg_bpp": jpeg_bpp,
        "bit_depth_bpp": bit_depth_bpp,
        "inference_time": inference_time,
    }


def main():
    args = get_args_parser()
    if args.config:
        config = builder.load_config(args.config)
    else:
        config = builder.load_config(builder.ckpt2config(args.ckpt))

    print("Config Path:", config.__name__)
    print("Use JPEG:", args.use_jpeg)
    args = builder.merge_config_args(config, args)
    torch.set_grad_enabled(False)

    model = args.model
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.prior_ic.update(force=True)

    model = model.to(device).eval()
    enc_model = EncWrapper(model)
    enc_model = (
        CustomDP(enc_model, device_ids=range(torch.cuda.device_count())) if torch.cuda.device_count() > 1 else enc_model
    )

    model = CustomDP(model, device_ids=range(torch.cuda.device_count())) if torch.cuda.device_count() > 1 else model

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if args.imgdir is None:
        return

    if isinstance(args.imgdir, str):
        args.imgdir = [args.imgdir]

    if not args.dryrun:
        for imgdir in args.imgdir:
            bpp = misc.AverageMeter()
            enc_time = misc.AverageMeter()
            dec_time = misc.AverageMeter()
            x_bpp = misc.AverageMeter()
            y_bpp = misc.AverageMeter()
            z_bpp = misc.AverageMeter()
            jpeg_bpp = misc.AverageMeter()
            bit_depth_bpp = misc.AverageMeter()

            if args.use_split:
                test_txt_path = os.path.join("./data", os.path.basename(imgdir), "test.txt")
                with open(test_txt_path, "r") as f:
                    paths = [line.strip() for line in f.readlines()]
            else:
                paths = os.listdir(imgdir)
            for path in tqdm(paths, desc=f"Processing {imgdir}"):
                raw_path = os.path.join(imgdir, path)

                cache_key = misc.get_md5(args.ckpt, raw_path, str(args.use_jpeg)) + ".json"
                if cache_key in os.listdir(args.cache_dir):
                    with open(os.path.join(args.cache_dir, cache_key), "r") as f:
                        enc_results = json.load(f)
                else:
                    *strings, enc_results = compress(enc_model, raw_path, args.use_jpeg)

                    with open(os.path.join(args.cache_dir, cache_key), "w") as f:
                        json.dump(enc_results, f)
                bpp.update(enc_results["bpp"])
                enc_time.update(enc_results["enc_time"])
                x_bpp.update(enc_results.get("x_bpp", 0))
                y_bpp.update(enc_results.get("y_bpp", 0))
                z_bpp.update(enc_results.get("z_bpp", 0))
                jpeg_bpp.update(enc_results.get("jpeg_bpp", 0))
                bit_depth_bpp.update(enc_results.get("bit_depth_bpp", 0))

            print(f"Results for {imgdir} using {args.ckpt}:")
            print(f"Average bpp: {bpp.avg:.4f}")
            print(f"Average Encoding Time: {enc_time.avg:.4f} seconds")
            print(f"Average Decoding Time: {dec_time.avg:.4f} seconds")
            print(
                f"x_bpp: {x_bpp.avg:.4f}, y_bpp: {y_bpp.avg:.4f}, z_bpp: {z_bpp.avg:.4f}, jpeg_bpp: {jpeg_bpp.avg:.4f}, bit_depth_bpp: {bit_depth_bpp.avg:.4f}"
            )
    else:
        # print("Dry run evaluate")

        for imgdir in args.imgdir:

            bpp = misc.AverageMeter()
            x_bpp = misc.AverageMeter()
            y_bpp = misc.AverageMeter()
            z_bpp = misc.AverageMeter()
            jpeg_bpp = misc.AverageMeter()
            bit_depth_bpp = misc.AverageMeter()
            infer_time = misc.AverageMeter()

            if args.use_split:
                test_txt_path = os.path.join("./data", os.path.basename(imgdir), "test.txt")
                with open(test_txt_path, "r") as f:
                    paths = [line.strip() for line in f.readlines()]
            else:
                paths = os.listdir(imgdir)
            for path in tqdm(paths):
                raw_path = os.path.join(imgdir, path)
                outs = dryrun(model, raw_path, args.use_jpeg)
                bpp.update(outs["loss"])
                x_bpp.update(outs.get("x_bpp", 0))
                y_bpp.update(outs.get("y_bpp", 0))
                z_bpp.update(outs.get("z_bpp", 0))
                jpeg_bpp.update(outs.get("jpeg_bpp", 0))
                bit_depth_bpp.update(outs.get("bit_depth_bpp", 0))
                infer_time.update(outs.get("inference_time", 0))

            print(f"Dry run Results for {imgdir} using {args.ckpt}:")
            print(f"Average bpp: {bpp.avg:.4f}")
            print(
                f"x_bpp: {x_bpp.avg:.4f}, y_bpp: {y_bpp.avg:.4f}, z_bpp: {z_bpp.avg:.4f}, jpeg_bpp: {jpeg_bpp.avg:.4f}, bit_depth_bpp: {bit_depth_bpp.avg:.4f}, inference_time: {infer_time.avg:.4f} seconds"
            )


if __name__ == "__main__":
    main()
