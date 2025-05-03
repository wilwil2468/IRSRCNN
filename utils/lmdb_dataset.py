import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    """
    A PyTorch Dataset wrapping an LMDB of (lr, hr) patches.
    Expects each record in LMDB to be:
      [6 bytes lr_shape][6 bytes hr_shape][lr_data][hr_data]
    where lr_shape = np.array(lr_np.shape, dtype=int16).tobytes(), etc.
    """

    def __init__(self, lmdb_path: str):
        # open in read-only, no locks, no readahead for speed
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        key = f"{index:08d}".encode()
        with self.env.begin(write=False) as txn:
            val = txn.get(key)

        # first 6 bytes: lr shape (3 × int16), next 6 bytes: hr shape
        lr_shape = np.frombuffer(val[0:6], dtype=np.int16)
        hr_shape = np.frombuffer(val[6:12], dtype=np.int16)

        lr_n = int(np.prod(lr_shape))
        hr_n = int(np.prod(hr_shape))

        # extract raw bytes and reshape
        offset = 12
        lr_np = np.frombuffer(val[offset:offset + lr_n], dtype=np.uint8).reshape(lr_shape)
        offset += lr_n
        hr_np = np.frombuffer(val[offset:offset + hr_n], dtype=np.uint8).reshape(hr_shape)

        # to CHW float tensors in [0,1]
        lr = torch.from_numpy(lr_np).permute(2, 0, 1).float().div(255.0)
        hr = torch.from_numpy(hr_np).permute(2, 0, 1).float().div(255.0)

        return lr, hr

    def get_batch(self, batch_size, shuffle_each_epoch=True):
        """
        Mimic the old interface:
        returns (lr_batch, hr_batch, isEnd_flag).
        """
        # Sample random indices
        idxs = torch.randint(0, len(self), (batch_size,), dtype=torch.long).tolist()
        lrs, hrs = [], []
        for i in idxs:
            lr, hr = self[i]        # calls __getitem__
            lrs.append(lr)
            hrs.append(hr)

        lr_batch = torch.stack(lrs, dim=0)
        hr_batch = torch.stack(hrs, dim=0)
        # We stream forever until steps are done, so never signal “epoch end”
        return lr_batch, hr_batch, False
