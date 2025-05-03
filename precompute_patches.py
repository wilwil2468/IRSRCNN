# precompute_patches_fast.py
import lmdb
import random
import argparse
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2     # pip install opencv-python
import os

def extract_patches(path, hr_size, lr_size, per_image):
    img = cv2.imread(str(path))                    # H×W×3, BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # to RGB
    H, W, _ = img.shape
    out = []
    for _ in range(per_image):
        x = random.randint(0, W - hr_size)
        y = random.randint(0, H - hr_size)
        hr = img[y:y+hr_size, x:x+hr_size]
        lr = cv2.resize(hr, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        out.append((lr.copy(), hr.copy()))
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    parser.add_argument("lmdb_path")
    parser.add_argument("--hr", type=int, default=21)
    parser.add_argument("--lr", type=int, default=33)
    parser.add_argument("--per-image", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(Path(args.src_dir).rglob("*.jpg"))
    TEN_GIB = 10 * 1024**3
    env = lmdb.open(
        args.lmdb_path,
        map_size=TEN_GIB,
        subdir=False,
        readonly=False,
        lock=True,
        writemap=True,     # memory‑map writes
        sync=False,        # skip fsync
        metasync=False
    )

    chunk_size = 5000
    idx = 0
    txn = env.begin(write=True)

    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(extract_patches, p, args.hr, args.lr, args.per_image): p
                   for p in paths}
        for fut in as_completed(futures):
            for lr_np, hr_np in fut.result():
                # pack meta + data
                meta = (np.array(lr_np.shape, dtype=np.int16).tobytes() +
                        np.array(hr_np.shape, dtype=np.int16).tobytes())
                val = meta + lr_np.tobytes() + hr_np.tobytes()
                key = f"{idx:08d}".encode()
                txn.put(key, val)
                idx += 1

                # commit in chunks
                if idx % chunk_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)

    txn.commit()
    env.close()
    print(f"Dumped {idx} patches to {args.lmdb_path}")
