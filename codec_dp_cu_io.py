import rawpy
import torch
import torch.nn.functional as F
from torchvision import transforms
import time
import utils.builder as builder
from utils.dist import CustomDP
from utils.preprocess import bayer_to_rggb, img2patch, patch2img
from datasets.transform import raw_to_tensor
import rawpy
import imagecodecs
import numpy as np
from model.codecwrapper_cu import EncWrapper, DecWrapper
import imageio.v2 as imageio
import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mix_num = 5
mix_num2 = mix_num * 2
mix_num3 = mix_num * 3

patch_sz = 64


def compress(model, raw_path, use_jpeg: bool):

    x_stream = []
    results = {}

    # read raw image
    raw = rawpy.imread(raw_path)

    # raw preprocess
    raw_img = raw.raw_image_visible if "NikonD40" not in raw_path else raw.raw_image
    hw = raw_img.shape[0] * raw_img.shape[1]
    raw_img = bayer_to_rggb(raw_img)
    img_shape = raw_img.shape[0:2]

    # render rgb image
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

    raw_img = raw_to_tensor(raw_img).unsqueeze(0)
    jpg = transforms.ToTensor()(jpg).unsqueeze(0)

    x = img2patch(raw_img, patch_sz=patch_sz).to(device)
    jpg_patch = img2patch(jpg, patch_sz=patch_sz).to(device)
    B = x.shape[0]
    # print("Batch size:", B)

    # torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        model.eval()
        outs = model(x, jpg_patch)

    # torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    results["enc_time"] = end_time - start_time
    if not torch.cuda.is_available() or torch.cuda.device_count() == 1:
        latent_code = outs[0]
        x_stream = outs[1]
        bit_depth = outs[2]
        y_len = sum(len(latent_code["strings"][i][0]) for i in range(len(latent_code["strings"])))
        z_len = len(latent_code["strings"][-1][0])

        x_len = sum([len(x_stream[i][0]) for i in range(len(x_stream))]) + len(x_stream)  # account for cnt
    else:
        latent_code = [out[0] for out in outs]
        x_stream = [out[1] for out in outs]
        bit_depth = [out[2] for out in outs]

        y_len = sum(sum(len(l["strings"][i][0]) for i in range(len(l["strings"]))) for l in latent_code)
        z_len = sum(len(l["strings"][-1][0]) for l in latent_code)

        x_len = sum(sum([len(xs[i][0]) for i in range(len(xs))]) for xs in x_stream) + sum(
            len(xs) for xs in x_stream
        )  # account for cnt

    results["z_bpp"] = z_len * 8 / hw
    results["y_bpp"] = y_len * 8 / hw
    results["x_bpp"] = x_len * 8 / hw

    results["latent_bpp"] = (y_len + z_len) * 8 / hw
    results["jpg_bpp"] = len(jpg_bin) * 8 / hw if use_jpeg else 0
    if torch.cuda.device_count() > 1:
        results["bit_depth_bpp"] = (
            B
            * model.module.codec.get_bit_depth_num()
            * torch.log2(torch.tensor(model.module.codec.end_bit - model.module.codec.start_bit + 1))
            / hw
        ).item()
    else:
        results["bit_depth_bpp"] = (
            B
            * model.codec.get_bit_depth_num()
            * torch.log2(torch.tensor(model.codec.end_bit - model.codec.start_bit + 1))
            / hw
        ).item()

    results["bpp"] = (
        results["latent_bpp"] + results["x_bpp"] + results["jpg_bpp"] + 6 * 2 * 8 / hw + results["bit_depth_bpp"]
    )  # latent stream + x stream + z_shape + x_shape

    return latent_code, x_stream, jpg_bin, bit_depth, img_shape, results


def decompress(model, strings):

    results = {}

    latent_code, x_stream, jpg_bin, bit_depth, img_shape = strings

    # render rgb image
    if len(jpg_bin) > 0:
        # jpg = imagecodecs.jpeg_decode(jpg_bin)
        # jpg = np.array(jpg, dtype=np.uint8)
        jpg = imageio.imread(io.BytesIO(jpg_bin))
    else:
        jpg = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

    jpg = transforms.ToTensor()(jpg).to(device).unsqueeze(0)

    # torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    jpg_patch = img2patch(jpg, patch_sz=patch_sz).to(device)

    if torch.cuda.device_count() > 1:
        jpg_patch = torch.chunk(jpg_patch, chunks=torch.cuda.device_count(), dim=0)
        bit_depth = [ch.to(device) for ch, device in zip(bit_depth, range(torch.cuda.device_count()))]
        jpg_patch = [ch.to(device) for ch, device in zip(jpg_patch, range(torch.cuda.device_count()))]
    with torch.no_grad():
        model.eval()
        # with misc.Timer(results, "pri decompress"):
        x_tmp = model(latent_code, x_stream, bit_depth, jpg_patch)
        if torch.cuda.device_count() > 1:
            outs = [out.to(device) for out in x_tmp]
            x_tmp = torch.cat(outs, dim=0)

    # torch.cuda.synchronize() if torch.cuda.is_available() else None
    results["dec_time"] = time.time() - start_time
    x = patch2img(x_tmp, img_shape)
    return x[0].cpu(), results


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="experiments/run-20251205-201722/checkpoints/best_model.pt",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument("--use_jpeg", action="store_true", help="Use JPEG as conditioning")
    parser.add_argument("--i", type=str, default="example/crop.tiff", help="Directory containing images to encode")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args_parser()
    if args.config:
        config = builder.load_config(args.config)
    else:
        config = builder.load_config(builder.ckpt2config(args.ckpt))
    args = builder.merge_config_args(config, args)
    torch.set_grad_enabled(False)

    model = args.model
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    model.prior_ic.update(force=True)

    model = model.to(device)
    enc_model = EncWrapper(model)
    enc_model = (
        CustomDP(enc_model, device_ids=range(torch.cuda.device_count())) if torch.cuda.device_count() > 1 else enc_model
    )

    import rawpy

    raw_image = rawpy.imread(args.i).raw_image_visible if "NikonD40" not in args.i else rawpy.imread(args.i).raw_image
    raw_image = bayer_to_rggb(raw_image)
    raw_image = raw_to_tensor(raw_image)
    print("Input raw image shape:", raw_image.shape)

    latent_code, x_stream, jpg_bin, bit_depth, img_shape, results = compress(enc_model, args.i, args.use_jpeg)

    print("Enc Results:", results)

    # decompress
    dec_model = DecWrapper(model)
    dec_model = (
        CustomDP(dec_model, device_ids=range(torch.cuda.device_count()), no_scatter=True)
        if torch.cuda.device_count() > 1
        else dec_model
    )
    img, results = decompress(dec_model, (latent_code, x_stream, jpg_bin, bit_depth, img_shape))
    print("Dec Results:", results)

    print("Decompressed image shape:", img.shape)

    # test right decompression
    print(torch.all(raw_image == img))

    # mse
    mse = F.mse_loss(raw_image.float(), img.float())
    print("mse:", mse.item())
