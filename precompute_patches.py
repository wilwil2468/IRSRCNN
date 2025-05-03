#!/usr/bin/env python3
import lmdb, random, argparse, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import numpy as np

def extract_patches(path, hr_size, lr_size, per_image, augment, master_seed):
    # reproducible per-image seed
    random.seed(master_seed ^ hash(str(path)))
    img = Image.open(path).convert('RGB')
    W, H = img.size
    if W < hr_size or H < hr_size:
        return []

    out = []
    for _ in range(per_image):
        # 1) random crop
        x = random.randint(0, W - hr_size)
        y = random.randint(0, H - hr_size)
        hr = img.crop((x, y, x + hr_size, y + hr_size))

        # 2) optional flips/rotations to match on-the-fly augment
        if augment:
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                hr = hr.rotate(angle)

        # 3) bicubic down‐sampling via PIL to ensure same kernel as original
        lr = hr.resize((lr_size, lr_size), Image.BICUBIC)

        # 4) to uint8 numpy
        hr_np = np.array(hr, dtype=np.uint8)
        lr_np = np.array(lr, dtype=np.uint8)

        out.append((lr_np, hr_np))
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precompute HR↔LR patches into LMDB with optional augmentation"
    )
    p.add_argument("src_dir",   help="root folder of source images (jpg/png)")
    p.add_argument("lmdb_path", help="output LMDB path")
    p.add_argument("--hr",       type=int, default=21,  help="HR crop size")
    p.add_argument("--lr",       type=int, default=33,  help="LR crop size")
    p.add_argument("--per-image",type=int, default=50,  help="patches per image")
    p.add_argument("--jobs",     type=int, default=4,   help="number of workers")
    p.add_argument("--augment",  action="store_true",   help="apply flips+rotations")
    p.add_argument("--seed",     type=int, default=None, help="master seed for RNG")
    args = p.parse_args()

    # pick a master seed if none given
    master_seed = args.seed if args.seed is not None else random.randrange(2**31)
    random.seed(master_seed)

    # gather all jpg/png under src_dir
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths += sorted(Path(args.src_dir).rglob(ext))

    # open LMDB
    TEN_GIB = 30 * 1024**3
    env = lmdb.open(
        args.lmdb_path,
        map_size=TEN_GIB,
        subdir=False,
        readonly=False,
        lock=True,
        writemap=True,
        sync=False,
        metasync=False,
    )

    chunk_size = 30_000
    idx = 0
    txn = env.begin(write=True)

    with ProcessPoolExecutor(
        max_workers=args.jobs
    ) as pool:
        futures = {
            pool.submit(
                extract_patches,
                path,
                args.hr,
                args.lr,
                args.per_image,
                args.augment,
                master_seed
            ): path
            for path in paths
        }

        for fut in as_completed(futures):
            patches = fut.result()
            for lr_np, hr_np in patches:
                # pack shapes + data
                meta = (
                    np.array(lr_np.shape, dtype=np.int16).tobytes()
                    + np.array(hr_np.shape, dtype=np.int16).tobytes()
                )
                val = meta + lr_np.tobytes() + hr_np.tobytes()
                key = f"{idx:08d}".encode("ascii")
                txn.put(key, val)
                idx += 1

                if idx % chunk_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)

    txn.commit()
    env.close()
    print(f"[+] Dumped {idx} patches → {args.lmdb_path}")
