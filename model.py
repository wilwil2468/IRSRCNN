import logging
from neuralnet import SRCNN_model
import numpy as np
import os
from utils.common import exists, tensor2numpy
import torch

# ——— simple logger setup ————————————————————————————————————————
def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

log = get_logger()
# ————————————————————————————————————————————————————————————————

class logger:
    def __init__(self, path, values) -> None:
        self.path = path
        self.values = values

class SRCNN:
    def __init__(self, architecture, device):
        self.device = device
        self.model = SRCNN_model(architecture).to(device)
        self.optimizer = None
        self.loss =  None
        self.metric = None
        self.model_path = None
        self.ckpt_path = None
        self.ckpt_man = None

    def setup(self, optimizer, loss, metric, model_path, ckpt_path):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        # @the best model weights
        self.model_path = model_path
        self.ckpt_path = ckpt_path

    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path):
            return
        self.ckpt_man = torch.load(ckpt_path, weights_only=True)
        self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
        self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        self.model.load_state_dict(
            torch.load(filepath,
                       map_location=torch.device(self.device),
                       weights_only=True
            )
        )

    def predict(self, lr):
        self.model.eval()
        with torch.no_grad():
            sr = self.model(lr)
        return sr

    def evaluate(self, dataset, batch_size=64):
        losses, metrics = [], []
        isEnd = False
        while not isEnd:
            lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.predict(lr)
            losses.append(tensor2numpy(self.loss(hr, sr)))
            metrics.append(tensor2numpy(self.metric(hr, sr)))

        return np.mean(losses), np.mean(metrics)

    def train(self, train_set, valid_set, batch_size, steps, save_every=1,
              save_best_only=False, save_log=False, log_dir=None):

        if save_log and log_dir is None:
            raise ValueError("log_dir must be specified if save_log is True")
        os.makedirs(log_dir, exist_ok=True)
        dict_logger = {
            "loss":       logger(path=os.path.join(log_dir, "losses.npy"),      values=[]),
            "metric":     logger(path=os.path.join(log_dir, "metrics.npy"),     values=[]),
            "val_loss":   logger(path=os.path.join(log_dir, "val_losses.npy"),  values=[]),
            "val_metric": logger(path=os.path.join(log_dir, "val_metrics.npy"), values=[])
        }
        for lg in dict_logger.values():
            if exists(lg.path):
                lg.values = np.load(lg.path).tolist()

        # resume / initialize
        cur_step = self.ckpt_man['step'] if self.ckpt_man else 0
        max_steps = cur_step + steps
        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_path)

        loss_buffer, metric_buffer = [], []
        log.info(f"Starting training for {steps} steps (from step {cur_step})")

        while cur_step < max_steps:
            cur_step += 1
            lr, hr, _ = train_set.get_batch(batch_size)
            loss, metric = self.train_step(lr, hr)
            loss_buffer.append(tensor2numpy(loss))
            metric_buffer.append(tensor2numpy(metric))

            # log every 100 steps
            if cur_step % 100 == 0:
                log.info(f"Progress step {cur_step}/{max_steps} — "
                         f"loss {np.mean(loss_buffer):.7f} — "
                         f"{self.metric.__name__} {np.mean(metric_buffer):.3f}")

            # periodic validation & checkpointing
            if (cur_step % save_every == 0) or (cur_step >= max_steps):
                avg_loss = np.mean(loss_buffer)
                avg_metric = np.mean(metric_buffer)
                val_loss, val_metric = self.evaluate(valid_set)
                log.info(f"Eval @ {cur_step}: "
                         f"train_loss {avg_loss:.7f} — "
                         f"train_{self.metric.__name__} {avg_metric:.3f} — "
                         f"val_loss {val_loss:.7f} — "
                         f"val_{self.metric.__name__} {val_metric:.3f}")

                if save_log:
                    dict_logger["loss"].values.append(avg_loss)
                    dict_logger["metric"].values.append(avg_metric)
                    dict_logger["val_loss"].values.append(val_loss)
                    dict_logger["val_metric"].values.append(val_metric)

                # save checkpoint
                torch.save({'step': cur_step,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()},
                           self.ckpt_path)

                # save best
                if not (save_best_only and val_loss > prev_loss):
                    prev_loss = val_loss
                    torch.save(self.model.state_dict(), self.model_path)
                    log.info(f"Saved best model at step {cur_step}")

                loss_buffer, metric_buffer = [], []

        # final log flush
        if save_log:
            for lg in dict_logger.values():
                np.save(lg.path, np.array(lg.values, dtype=np.float32))
        log.info("Training complete.")

    def train_step(self, lr, hr):
        self.model.train()
        self.optimizer.zero_grad()
        lr, hr = lr.to(self.device), hr.to(self.device)
        sr = self.model(lr)
        loss = self.loss(hr, sr)
        metric = self.metric(hr, sr)
        loss.backward()
        self.optimizer.step()
        return loss, metric
