from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
import argparse
import os
import numpy as np
from model import SRCNN
from utils.common import PSNR
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

import random
from pathlib import Path
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import lmdb

class LMDBPatchDataset(Dataset):
    def __init__(self, lmdb_path):
        # readonly, lock=False speeds up multiworker reads
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with self.env.begin() as txn:
            # count entries
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = f"{idx:08d}".encode()
        with self.env.begin() as txn:
            data = txn.get(key)
        # unpack meta + raw bytes
        # first 3 int16 for lr shape, next 3 int16 for hr shape
        meta_bytes = 6 * 2  # 6 int16 values
        meta = data[:meta_bytes]
        shapes = np.frombuffer(meta, dtype=np.int16)
        lr_shape = tuple(shapes[:3])
        hr_shape = tuple(shapes[3:6])
        raw = data[meta_bytes:]
        # split raw into lr and hr
        lr_n = np.prod(lr_shape)
        lr_np = np.frombuffer(raw[:lr_n], dtype=np.uint8).reshape(lr_shape)
        hr_np = np.frombuffer(raw[lr_n:], dtype=np.uint8).reshape(hr_shape)
        # to torch, [C,H,W], float32
        lr = torch.from_numpy(lr_np).permute(2,0,1).float().div(255)
        hr = torch.from_numpy(hr_np).permute(2,0,1).float().div(255)
        return lr, hr


# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000)
parser.add_argument("--batch-size",     type=int, default=128)
parser.add_argument("--architecture",   type=str, default="915")
parser.add_argument("--save-every",     type=int, default=1000)
parser.add_argument("--save-log",       type=int, default=0)
parser.add_argument("--save-best-only", type=int, default=0)
parser.add_argument("--ckpt-dir",       type=str, default="")
parser.add_argument("--num-worker",     type=int, default=4)
FLAGS, _ = parser.parse_known_args()

# ── Hyperparameters & paths ────────────────────────────────────────────────────
steps          = FLAGS.steps
batch_size     = FLAGS.batch_size
save_every     = FLAGS.save_every
save_log       = (FLAGS.save_log == 1)
save_best_only = (FLAGS.save_best_only == 1)
architecture   = FLAGS.architecture
num_worker   = FLAGS.num_worker

if architecture not in ["915","935","955"]:
    raise ValueError("architecture must be 915, 935, or 955")

ckpt_dir = FLAGS.ckpt_dir or f"checkpoint/SRCNN{architecture}"
os.makedirs(ckpt_dir, exist_ok=True)
model_path = os.path.join(ckpt_dir, f"SRCNN-{architecture}.pt")
ckpt_path  = os.path.join(ckpt_dir, "ckpt.pt")

dataset_dir   = "dataset"
lr_crop_size  = 33
hr_crop_size  = 21 if architecture=="915" else 19 if architecture=="935" else 17

# ── 1) Create DataLoaders ────────────────────────────────────────────────────
train_ds = LMDBPatchDataset(f"train_patches_{architecture}.lmdb")
valid_ds = LMDBPatchDataset(f"valid_patches_{architecture}.lmdb")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_worker, pin_memory=True, persistent_workers=True, prefetch_factor=4
)
valid_loader = DataLoader(
    valid_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_worker, pin_memory=True, persistent_workers=True, prefetch_factor=4
)

# ── 2) Wrap them into get_batch API ──────────────────────────────────────────
class BatchLoaderWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None

    # accept shuffle_each_epoch (model.evaluate passes it) but ignore,
    # relying on DataLoader(shuffle=True) to reshuffle when you re-iter
    def get_batch(self, _batch_size, shuffle_each_epoch: bool = False):
        if self.iterator is None:
           self.iterator = iter(self.loader)
        try:
            lr, hr = next(self.iterator)
        except StopIteration:
            # end of epoch: re-create iterator (DataLoader will reshuffle if shuffle=True)
            self.iterator = iter(self.loader)
            lr, hr = next(self.iterator)
        return lr, hr, False

train_set = BatchLoaderWrapper(train_loader)
valid_set = BatchLoaderWrapper(valid_loader)

# ── 3) Model setup & training ─────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    srcnn = SRCNN(architecture, device)
    srcnn.setup(
        optimizer=torch.optim.Adam(srcnn.model.parameters(), lr=2e-5),
        loss=torch.nn.MSELoss(),
        model_path=model_path,
        ckpt_path=ckpt_path,
        metric=PSNR
    )
    srcnn.load_checkpoint(ckpt_path)
    srcnn.train(
        train_set,     # now has get_batch()
        valid_set,     # now has get_batch()
        steps=steps,
        batch_size=batch_size,
        save_best_only=save_best_only,
        save_every=save_every,
        save_log=save_log,
        log_dir=ckpt_dir
    )

if __name__ == "__main__":
    main()
