# precompute_patches.py
import lmdb
import random
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as TF

def make_patches(img, hr_size, lr_size, num_patches):
    # img: PIL.Image RGB
    W, H = img.size
    patches = []
    for _ in range(num_patches):
        x = random.randint(0, W - hr_size)
        y = random.randint(0, H - hr_size)
        hr = img.crop((x, y, x + hr_size, y + hr_size))
        lr = hr.resize((lr_size, lr_size), Image.BICUBIC)
        # convert to uint8 np arrays
        hr_np = np.array(hr, dtype=np.uint8)  # H×W×3
        lr_np = np.array(lr, dtype=np.uint8)
        patches.append((lr_np, hr_np))
    return patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", help="folder of .jpg images")
    parser.add_argument("lmdb_path", help="where to write LMDB")
    parser.add_argument("--hr", type=int, default=21)
    parser.add_argument("--lr", type=int, default=33)
    parser.add_argument("--per-image", type=int, default=10,
                        help="how many random patches per source image")
    args = parser.parse_args()

    paths = sorted(Path(args.src_dir).rglob("*.jpg"))
    total_patches = len(paths) * args.per_image

    # compute bytes per patch exactly (HR + LR + meta)
    hr_bytes = args.hr * args.hr * 3
    lr_bytes = args.lr * args.lr * 3
    meta_bytes = 6 * 2                # six int16 values
    patch_bytes = hr_bytes + lr_bytes + meta_bytes
    # total with 10% slack
    map_size = int(patch_bytes * total_patches * 1.10)
    print(f"Setting LMDB map_size = {map_size / (1024**3):.2f} GB")
    env = lmdb.open(args.lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        idx = 0
        for p in paths:
            img = Image.open(p).convert("RGB")
            for lr_np, hr_np in make_patches(
                    img, args.hr, args.lr, args.per_image):
                # key must be bytes
                key = f"{idx:08d}".encode()
                # value: pack both arrays with np.save into bytes
                # we prefix with lr size so we can split
                lr_bytes = lr_np.tobytes()
                hr_bytes = hr_np.tobytes()
                # store shape in metadata
                meta = np.array(lr_np.shape, dtype=np.int16).tobytes() + \
                       np.array(hr_np.shape, dtype=np.int16).tobytes()
                txn.put(key, meta + lr_bytes + hr_bytes)
                idx += 1
            print(f"Image {p.name} → patches {idx}")
    env.close()
    print(f"Dumped {idx} patches to {args.lmdb_path}")
