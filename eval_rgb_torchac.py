import torch
from PIL import Image
from torchvision import transforms
import os
from codec_rgb_torchac import compress, decompress
from model.codecwrapper_cu_rgb_torchac import EncWrapper, DecWrapper
import utils.misc as misc
from utils.preprocess import img2patch
import utils.builder as builder
from model.loss import BPPLoss
from utils.dist import CustomDP
from tqdm import tqdm

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
    parser.add_argument("--cache_dir", type=str, default=".cache_rgb", help="Directory to cache results")
    # parser.add_argument("--dryrun", action="store_true", help="Run dry run for testing purposes")
    return parser.parse_args()


def main():
    args = get_args_parser()
    if args.config:
        config = builder.load_config(args.config)
    else:
        config = builder.load_config(builder.ckpt2config(args.ckpt))

    print("Config Path:", config.__name__)

    args = builder.merge_config_args(config, args)
    torch.set_grad_enabled(False)

    args.model.load_state_dict(torch.load(args.ckpt)["model"])

    args.model.prior_ic.update(force=True)

    args.model = args.model.to(device).eval()
    model = args.model
    enc_model = EncWrapper(model)

    enc_model = (
        CustomDP(enc_model, device_ids=range(torch.cuda.device_count())) if torch.cuda.device_count() > 1 else enc_model
    )

    dec_model = DecWrapper(model)
    dec_model = (
        CustomDP(dec_model, device_ids=range(torch.cuda.device_count()), no_scatter=True)
        if torch.cuda.device_count() > 1
        else dec_model
    )
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if args.imgdir is None:
        return

    if isinstance(args.imgdir, str):
        args.imgdir = [args.imgdir]

    for imgdir in args.imgdir:
        bpp = misc.AverageMeter()
        enc_time = misc.AverageMeter()
        dec_time = misc.AverageMeter()

        for path in tqdm(os.listdir(imgdir)):
            img_path = os.path.join(imgdir, path)

            cache_key = misc.get_md5(args.ckpt, img_path)
            if cache_key in os.listdir(args.cache_dir):
                enc_results, dec_results = torch.load(os.path.join(args.cache_dir, cache_key))
            else:
                *strings, enc_results = compress(enc_model, img_path, None)
                img, dec_results = decompress(dec_model, strings)
                original_img = Image.open(img_path).convert("RGB")
                original_img = transforms.PILToTensor()(original_img)

                if not torch.all(img == original_img):
                    print(f"Warning: Decoded image does not match the original image for {img_path}")
                torch.save((enc_results, dec_results), os.path.join(args.cache_dir, cache_key))
            bpp.update(enc_results["bpp"])
            enc_time.update(enc_results["enc_time"])
            dec_time.update(dec_results["dec_time"])

        print(f"Results for {imgdir} using {args.ckpt}:")
        print(f"Average BPP: {bpp.avg:.2f}")
        print(f"Average Encoding Time: {enc_time.avg:.2f} seconds")
        print(f"Average Decoding Time: {dec_time.avg:.2f} seconds")


if __name__ == "__main__":
    main()
