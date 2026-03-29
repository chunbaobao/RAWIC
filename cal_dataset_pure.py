import os
import rawpy
import numpy as np
import pandas as pd
from PIL import Image
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import exifread

base_dir = "/NEW_EDS/JJ_Group/zhengch2506/datasets/nus8"


def compute_mse(img1, img2):

    img1 = np.array(img1, dtype=np.float32)
    img2 = np.array(img2, dtype=np.float32)
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    return np.mean((img1 - img2) ** 2)


def bytes_to_mb(x):

    return round(x / (1024 * 1024), 2)


def process_one_file(args):

    camera, raw_path = args
    name, ext = os.path.splitext(os.path.basename(raw_path))

    try:
        raw_size = os.path.getsize(raw_path)

        with rawpy.imread(raw_path) as raw:
            bit_depth = raw.raw_image.dtype.itemsize * 8
            raw_shape = raw.raw_image.shape
            raw_visible_shape = raw.raw_image_visible.shape

            thumb = raw.extract_thumb()
            thumb_size = len(thumb.data)
            if thumb.format == rawpy.ThumbFormat.JPEG:
                thumb_img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                thumb_img = Image.fromarray(thumb.data)
            else:
                thumb_img = None

            thumb_shape = None
            if thumb_img is not None:
                thumb_shape = thumb_img.size[::-1]
            max_value = raw.raw_image_visible.max()
            min_value = raw.raw_image_visible.min()
            white_level = raw.white_level
            black_level = raw.black_level_per_channel
        with open(raw_path, "rb") as f:
            tags = exifread.process_file(f)
            orientation_tag = tags.get("Image Orientation")

        return {
            "Camera": camera,
            "Filename": name + ext,
            "RAW_Size_MB": bytes_to_mb(raw_size),
            "RAW_Shape": str(raw_shape),
            "RAW_Visible_Shape": str(raw_visible_shape),
            "Thumbnail_Size_MB": bytes_to_mb(thumb_size),
            "Thumbnail_Shape": str(thumb_shape),
            "Bit_Depth": bit_depth,
            "ROTATE": str(orientation_tag) if orientation_tag else "N/A",
            "MAX_VALUE": max_value,
            "MIN_VALUE": min_value,
            "WHITE_LEVEL": white_level,
            "BLACK_LEVELs": str(black_level),
        }

    except Exception as e:
        print(f"[ERROR] {camera}/{name}{ext} -> {e}")
        return None


def main():
    tasks = []
    for camera in os.listdir(base_dir):
        camera_path = os.path.join(base_dir, camera)
        raw_dir = camera_path

        raw_files = [f for f in os.listdir(raw_dir) if not f.startswith(".")]
        for raw_file in raw_files:
            raw_path = os.path.join(raw_dir, raw_file)
            tasks.append((camera, raw_path))

    print(f"📸 Total {len(tasks)} RAW files, starting parallel processing...")

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_one_file, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            res = f.result()
            if res is not None:
                results.append(res)

    results.sort(key=lambda x: (x["Camera"], x["Filename"]))

    df = pd.DataFrame(results)
    df.to_excel("nus8_pure_stats.xlsx", index=False)
    print("Saved as nus8_pure_stats.xlsx")


if __name__ == "__main__":
    main()
