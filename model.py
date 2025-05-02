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
        # instantiate and push to device
        self.model = SRCNN_model(architecture).to(device)
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
        if not os.path.exists(ckpt_path):
            return
        self.ckpt_man = torch.load(ckpt_path)
        self.optimizer.load_state_dict(self.ckpt_man['optimizer'])
        self.model.load_state_dict(self.ckpt_man['model'])

    def load_weights(self, filepath):
        self.model.load_state_dict(
            torch.load(filepath, map_location=torch.device(self.device))
        )

    def predict(self, lr_batch):
        self.model.eval()
        return self.model(lr_batch)

    def evaluate(self, dataloader):
        losses, metrics = [], []
        for lr, hr in dataloader:
            lr, hr = lr.to(self.device), hr.to(self.device)
            with torch.no_grad():
                sr = self.predict(lr)
            loss = self.loss(hr, sr)
            metric = self.metric(hr, sr)
            losses.append(tensor2numpy(loss))
            metrics.append(tensor2numpy(metric))
        return np.mean(losses), np.mean(metrics)

    def train(self, train_loader, valid_loader, batch_size, steps,
              save_every=1, save_best_only=False, save_log=False, log_dir=None):

        if save_log and log_dir is None:
            raise ValueError("log_dir must be specified if save_log is True")
        os.makedirs(log_dir, exist_ok=True)
        dict_logger = {
            "loss":       logger(os.path.join(log_dir, "losses.npy"), []),
            "metric":     logger(os.path.join(log_dir, "metrics.npy"), []),
            "val_loss":   logger(os.path.join(log_dir, "val_losses.npy"), []),
            "val_metric": logger(os.path.join(log_dir, "val_metrics.npy"), [])
        }
        for lg in dict_logger.values():
            if exists(lg.path):
                lg.values = np.load(lg.path).tolist()

        start_step = self.ckpt_man['step'] if self.ckpt_man else 0
        best_loss = np.inf
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            best_loss, _ = self.evaluate(valid_loader)
            self.load_checkpoint(self.ckpt_path)

        step = 0
        loss_buffer, metric_buffer = [], []
        for lr, hr in train_loader:
            if step >= steps:
                break
            step += 1
            lr, hr = lr.to(self.device), hr.to(self.device)
            loss, metric = self.train_step(lr, hr)
            loss_buffer.append(tensor2numpy(loss))
            metric_buffer.append(tensor2numpy(metric))

            if step % (save_every // 10 or 1) == 0:
                log.info(f"Progress step {step}/{steps} - loss {np.mean(loss_buffer):.4f} - {self.metric.__name__} {np.mean(metric_buffer):.4f}")

            if step % save_every == 0 or step == steps:
                val_loss, val_metric = self.evaluate(valid_loader)
                log.info(f"Eval at step {step}: val_loss {val_loss:.4f} val_{self.metric.__name__} {val_metric:.4f}")

                if save_log:
                    dict_logger["loss"].values.append(np.mean(loss_buffer))
                    dict_logger["metric"].values.append(np.mean(metric_buffer))
                    dict_logger["val_loss"].values.append(val_loss)
                    dict_logger["val_metric"].values.append(val_metric)

                torch.save({'step': step, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, self.ckpt_path)
                if not (save_best_only and val_loss > best_loss):
                    best_loss = val_loss
                    torch.save(self.model.state_dict(), self.model_path)
                    log.info(f"Saved best model at step {step}\n")
                loss_buffer, metric_buffer = [], []

        if save_log:
            for lg in dict_logger.values():
                np.save(lg.path, np.array(lg.values, dtype=np.float32))

    def train_step(self, lr, hr):
        self.model.train()
        self.optimizer.zero_grad()
        sr = self.model(lr)
        loss = self.loss(hr, sr)
        metric = self.metric(hr, sr)
        loss.backward()
        self.optimizer.step()
        return loss, metric