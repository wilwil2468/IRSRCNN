from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

from model import SRCNN
from utils.common import PSNR

import argparse
import torch
import os

# ── On‑the‑fly patch dataset ──────────────────────────────────────────────────
class OnTheFlyPatchDataset(Dataset):
    def __init__(self, root_dir, lr_size, hr_size):
        self.paths = sorted(Path(root_dir).rglob("*.jpg"))
        self.lr_size, self.hr_size = lr_size, hr_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        # random HR crop
        x = random.randint(0, img.width  - self.hr_size)
        y = random.randint(0, img.height - self.hr_size)
        hr = img.crop((x, y, x + self.hr_size, y + self.hr_size))
        lr = hr.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        # convert to tensor and normalize [0,1]
        to_tensor = torch.nn.functional.interpolate  # replace with your actual to_tensor
        return to_tensor(torch.tensor(lr).permute(2,0,1).unsqueeze(0), size=None), \
               to_tensor(torch.tensor(hr).permute(2,0,1).unsqueeze(0), size=None)

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000)
parser.add_argument("--batch-size",     type=int, default=128)
parser.add_argument("--architecture",   type=str, default="915")
parser.add_argument("--save-every",     type=int, default=1000)
parser.add_argument("--save-log",       type=int, default=0)
parser.add_argument("--save-best-only", type=int, default=0)
parser.add_argument("--ckpt-dir",       type=str, default="")
FLAGS, _ = parser.parse_known_args()

# ── Hyperparameters & paths ────────────────────────────────────────────────────
steps          = FLAGS.steps
batch_size     = FLAGS.batch_size
save_every     = FLAGS.save_every
save_log       = (FLAGS.save_log == 1)
save_best_only = (FLAGS.save_best_only == 1)
architecture   = FLAGS.architecture

if architecture not in ["915","935","955"]:
    raise ValueError("architecture must be 915, 935, or 955")

ckpt_dir = FLAGS.ckpt_dir or f"checkpoint/SRCNN{architecture}"
os.makedirs(ckpt_dir, exist_ok=True)
model_path = os.path.join(ckpt_dir, f"SRCNN-{architecture}.pt")
ckpt_path  = os.path.join(ckpt_dir, "ckpt.pt")

dataset_dir   = "dataset"
lr_crop_size  = 33
hr_crop_size  = 21 if architecture=="915" else 19 if architecture=="935" else 17

# ── Prepare DataLoaders ───────────────────────────────────────────────────────
train_ds  = OnTheFlyPatchDataset(os.path.join(dataset_dir,"train"),      lr_crop_size, hr_crop_size)
valid_ds  = OnTheFlyPatchDataset(os.path.join(dataset_dir,"validation"), lr_crop_size, hr_crop_size)

train_loader = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ── Training ───────────────────────────────────────────────────────────────────
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
        train_loader,        # now a DataLoader
        valid_loader,        # now a DataLoader
        steps=steps,
        batch_size=None,     # batch_size is baked into the loader
        save_best_only=save_best_only,
        save_every=save_every,
        save_log=save_log,
        log_dir=ckpt_dir
    )

if __name__ == "__main__":
    main()