from neuralnet import SRCNN_model
import numpy as np
import os
from utils.common import exists, tensor2numpy
import torch
import logging

# configure a simple logger
def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

log = get_logger()

class logger:
    def __init__(self, path, values) -> None:
        self.path = path
        self.values = values

class SRCNN:
    def __init__(self, architecture, device):
        self.device = device
        # instantiate and push to CUDA
        self.model = SRCNN_model(architecture).to(device)
        # if you’re on PyTorch ≥2.0, compile it into a single graph
        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.optimizer = None
        self.loss = None
        self.metric = None
        self.model_path = None
        self.ckpt_path = None
        self.ckpt_man = None

    def setup(self, optimizer, loss, metric, model_path, ckpt_path):
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.model_path = model_path
        self.ckpt_path = ckpt_path

    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path):
            return
        self.ckpt_man = torch.load(ckpt_path)
        self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
        self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=torch.device(self.device)))

    def predict(self, lr):
        self.model.eval()
        sr = self.model(lr)
        return sr

    def evaluate(self, dataset, batch_size=64):
        losses, metrics = [], []
        isEnd = False
        while not isEnd:
            lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.predict(lr)
            loss = self.loss(hr, sr)
            metric = self.metric(hr, sr)
            losses.append(tensor2numpy(loss))
            metrics.append(tensor2numpy(metric))

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
        for key, lg in dict_logger.items():
            if exists(lg.path):
                lg.values = np.load(lg.path).tolist()

        cur_step = self.ckpt_man['step'] if self.ckpt_man is not None else 0
        max_steps = cur_step + steps

        prev_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_path)

        log_interval = max(1, save_every // 10)
        loss_buffer = []
        metric_buffer = []

        while cur_step < max_steps:
            cur_step += 1
            lr, hr, _ = train_set.get_batch(batch_size)
            loss, metric = self.train_step(lr, hr)
            loss_buffer.append(tensor2numpy(loss))
            metric_buffer.append(tensor2numpy(metric))

            # log intermediate progress
            if cur_step % log_interval == 0:
                avg_loss = np.mean(loss_buffer)
                avg_metric = np.mean(metric_buffer)
                log.info(f"Progress: Step {cur_step}/{max_steps} — loss: {avg_loss:.7f} — {self.metric.__name__}: {avg_metric:.3f}")

            if (cur_step % save_every == 0) or (cur_step >= max_steps):
                loss_mean = np.mean(loss_buffer)
                metric_mean = np.mean(metric_buffer)
                val_loss, val_metric = self.evaluate(valid_set)

                log.info(
                    f"Step {cur_step}/{max_steps} — loss: {loss_mean:.7f} — "
                    f"{self.metric.__name__}: {metric_mean:.3f} — "
                    f"val_loss: {val_loss:.7f} — val_{self.metric.__name__}: {val_metric:.3f}"
                )

                if save_log:
                    dict_logger["loss"].values.append(loss_mean)
                    dict_logger["metric"].values.append(metric_mean)
                    dict_logger["val_loss"].values.append(val_loss)
                    dict_logger["val_metric"].values.append(val_metric)

                loss_buffer = []
                metric_buffer = []
                torch.save({
                    'step': cur_step,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, self.ckpt_path)

                if not (save_best_only and val_loss > prev_loss):
                    prev_loss = val_loss
                    self.model.eval()
                    torch.save(self.model.state_dict(), self.model_path)
                    log.info(f"Saved model to {self.model_path}\n")

        if save_log:
            for key, lg in dict_logger.items():
                np.save(lg.path, np.array(lg.values, dtype=np.float32))

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
