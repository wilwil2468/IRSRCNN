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

class OnTheFlyPatchDataset(Dataset):
    def __init__(self, root_dir, lr_size, hr_size):
        self.paths = sorted(Path(root_dir).rglob("*.png"))
        if len(self.paths) == 0:
            raise RuntimeError(f"No .png files found in '{root_dir}'")
        self.lr_size, self.hr_size = lr_size, hr_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = random.randint(0, img.width  - self.hr_size)
        y = random.randint(0, img.height - self.hr_size)
        hr = img.crop((x, y, x + self.hr_size, y + self.hr_size))
        lr = hr.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        lr_t = torch.from_numpy(np.array(lr)).permute(2,0,1).float().div(255)
        hr_t = torch.from_numpy(np.array(hr)).permute(2,0,1).float().div(255)
        return lr_t, hr_t

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",          type=int,   default=100000)
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--architecture",   type=str,   default="915")
    parser.add_argument("--save-every",     type=int,   default=1000)
    parser.add_argument("--save-log",       action="store_true")
    parser.add_argument("--save-best-only", action="store_true")
    parser.add_argument("--ckpt-dir",       type=str,   default="")
    return parser.parse_args()

def main():
    FLAGS = parse_args()

    # validate architecture
    if FLAGS.architecture not in ["915","935","955"]:
        raise ValueError("architecture must be one of 915, 935, 955")

    # prepare checkpoint dir
    ckpt_dir = FLAGS.ckpt_dir or f"checkpoint/SRCNN{FLAGS.architecture}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model_path = os.path.join(ckpt_dir, f"SRCNN-{FLAGS.architecture}.pt")
    ckpt_path  = os.path.join(ckpt_dir, "ckpt.pt")

    # determine crop sizes
    lr_crop_size = 33
    hr_crop_size = {"915":21, "935":19, "955":17}[FLAGS.architecture]

    # create datasets & loaders
    train_ds = OnTheFlyPatchDataset("dataset/train",      lr_crop_size, hr_crop_size)
    valid_ds = OnTheFlyPatchDataset("dataset/validation", lr_crop_size, hr_crop_size)

    train_loader = DataLoader(train_ds, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=FLAGS.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # wrap to get_batch API
    class BatchLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
            self.iterator = iter(loader)
        def get_batch(self, _batch_size):
            try:
                return next(self.iterator) + (None,)
            except StopIteration:
                self.iterator = iter(self.loader)
                return next(self.iterator) + (None,)

    train_set = BatchLoaderWrapper(train_loader)
    valid_set = BatchLoaderWrapper(valid_loader)

    # model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    srcnn = SRCNN(FLAGS.architecture, device)
    srcnn.setup(
        optimizer    = torch.optim.Adam(srcnn.model.parameters(), lr=2e-5),
        loss         = torch.nn.MSELoss(),
        model_path   = model_path,
        ckpt_path    = ckpt_path,
        metric       = PSNR
    )
    srcnn.load_checkpoint(ckpt_path)
    srcnn.train(
        train_set,
        valid_set,
        steps         = FLAGS.steps,
        batch_size    = FLAGS.batch_size,
        save_best_only= FLAGS.save_best_only,
        save_every    = FLAGS.save_every,
        save_log      = FLAGS.save_log,
        log_dir       = ckpt_dir
    )

if __name__ == "__main__":
    main()
