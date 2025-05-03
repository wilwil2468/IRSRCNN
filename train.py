from utils.dataset import dataset
 from utils.common import PSNR
 from model import SRCNN
from utils.lmdb_dataset import LMDBDataset
 import argparse
import torch.utils.data as data
 import torch
 import os

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

# -----------------------------------------------------------
#  Init datasets (either from LMDB or on-the-fly)
# -----------------------------------------------------------

# paths where we expect precomputed patches
train_lmdb = f"train_patches_{architecture}.lmdb"
valid_lmdb = f"valid_patches_{architecture}.lmdb"

# if both LMDB paths are provided, use the precomputed patches
if os.path.exists(train_lmdb) and os.path.exists(valid_lmdb):
    print(f"→ Loading train patches from LMDB: {train_lmdb}")
    train_set = LMDBDataset(train_lmdb)
    print(f"→ Loading valid patches from LMDB: {valid_lmdb}")
    valid_set = LMDBDataset(valid_lmdb)

else:
    # fallback to on-the-fly patch generation
    print("PRECOMPUTED PATCHES NOT FOUND!\nFalling back to on‑the‑fly generation.")
    dataset_dir = "dataset"
    lr_crop_size = 33
    hr_crop_size = 21
    if architecture == "935":
        hr_crop_size = 19
    elif architecture == "955":
        hr_crop_size = 17

    train_set = dataset(dataset_dir, "train")
    train_set.generate(lr_crop_size, hr_crop_size)
    train_set.load_data()

    valid_set = dataset(dataset_dir, "validation")
    valid_set.generate(lr_crop_size, hr_crop_size)
    valid_set.load_data()

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
        train_loader,     # now has get_batch()
        valid_loader,     # now has get_batch()
        steps=steps,
        batch_size=batch_size,
        save_best_only=save_best_only,
        save_every=save_every,
        save_log=save_log,
        log_dir=ckpt_dir
    )

if __name__ == "__main__":
    main()
