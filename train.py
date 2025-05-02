from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torch
import argparse
import os
from model import SRCNN
from utils.common import PSNR
from torchvision import transforms
import kornia.augmentation as K
from kornia.constants import Resample

# ── GPU Patch Dataset ──────────────────────────────────────────────────

class GPUPatchDataset(Dataset):
    def __init__(self, root_dir, lr_size, hr_size, device):
        self.device = device
        self.hr_size = hr_size
        self.lr_size = lr_size

        # 1) preload all images into RAM
        self.raw_imgs = [
            Image.open(p).convert("RGB")
            for p in sorted(Path(root_dir).rglob("*.jpg"))
        ]

        # 2) CPU → Tensor
        self.to_tensor = transforms.ToTensor()

        # 3) GPU‐side Kornia transforms:
        #    RandomCrop → [1,C,hr,hr]
        #    Resize    → [1,C,lr,lr]
        self.crop   = K.RandomCrop((hr_size, hr_size)).to(device)
        self.resize = K.Resize(
            (lr_size, lr_size),
            resample=Resample.BICUBIC.name,   # <-- use `resample=` here
            align_corners=True,               # optional, controls corner alignment
            antialias=True                    # optional, adds Gaussian pre-filtering when downscaling
        ).to(device)

    def __len__(self):
        return len(self.raw_imgs)

    def __getitem__(self, idx):
        # (a) CPU: PIL → Tensor
        img = self.raw_imgs[idx]
        img_t = self.to_tensor(img).unsqueeze(0)  # [1,3,H,W]

        # (b) → GPU + crop & resize
        img_t = img_t.to(self.device, non_blocking=True)
        hr_patch = self.crop(img_t)               # [1,3,hr,hr]
        lr_patch = self.resize(hr_patch)          # [1,3,lr,lr]

        # (c) squeeze batch dim and return
        return lr_patch.squeeze(0), hr_patch.squeeze(0)

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = GPUPatchDataset("dataset/train",      lr_crop_size, hr_crop_size, device)
valid_ds = GPUPatchDataset("dataset/validation", lr_crop_size, hr_crop_size, device)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_worker, pin_memory=False, persistent_workers=True, prefetch_factor=2)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=False, persistent_workers=True, prefetch_factor=2)

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
