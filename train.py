from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import os
import numpy as np
from model import SRCNN
from utils.common import PSNR
from torchvision import transforms

import random
from pathlib import Path
from functools import lru_cache

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms.functional as TF

device = "cuda"

# ── On‑the‑fly patch dataset ──────────────────────────────────────────────────
class CachedPatchDataset(Dataset):
    def __init__(self, root_dir, lr_size, hr_size, cache_size=2048):
        self.paths   = sorted(Path(root_dir).rglob("*.jpg"))
        self.lr_size = lr_size
        self.hr_size = hr_size
        # decorate the loader to keep the N most-recent images in RAM
        self._load_image = lru_cache(maxsize=cache_size)(self._load_image_uncached)

    def __len__(self):
        return len(self.paths)

    def _load_image_uncached(self, path_str):
        # fast JPEG → torch.Tensor, [C,H,W], float in [0,1]
        return read_image(path_str).float().div(255)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        img = read_image(path).float().div(255).unsqueeze(0).to(device)  # [1,C,H,W]

        hr  = self.crop(img)                                             # Kornia RandomCrop on GPU
        lr  = self.resize(hr)                                            # Kornia Resize on GPU

        return lr.squeeze(0), hr.squeeze(0)

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
train_ds = CachedPatchDataset("dataset/train",      lr_crop_size, hr_crop_size)
valid_ds = CachedPatchDataset("dataset/validation", lr_crop_size, hr_crop_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_worker, pin_memory=True, persistent_workers=True, prefetch_factor=2)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True, persistent_workers=True, prefetch_factor=2)

# ── 2) Wrap them into get_batch API ──────────────────────────────────────────
class BatchLoaderWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
    def get_batch(self, _batch_size):
        try:
            lr, hr = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            lr, hr = next(self.iterator)
        return lr, hr, None

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
