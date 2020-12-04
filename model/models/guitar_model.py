import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
import pickle

from ..utils import error_to_signal
from .wavenet import WNet


class GuitarModel(pl.LightningModule):
    def __init__(self, hparams):
        super(GuitarModel, self).__init__()
        print(hparams)
        self.WNet = WNet(
            num_channels=hparams.num_channels,
            dilation_depth=hparams.dilation_depth,
            num_repeat=hparams.num_repeat,
            kernel_size=hparams.kernel_size,
        )
        self.hparams = hparams

    def forward(self, x):
        return self.WNet(x)

    def training_step(self, batch):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2):], y_pred).mean()
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch):
        x, y = batch
        y_pred = self.forward(x)
        loss = error_to_signal(y[:, :, -y_pred.size(2):], y_pred).mean()
        return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

    def prepare_data(self):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        data = pickle.load(open(self.hparams.data, "rb"))
        self.train_ds = ds(data["x_train"], data["y_train"])
        self.valid_ds = ds(data["x_valid"], data["y_valid"])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.WNet.parameters(), lr=self.hparams.learning_rate
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.hparams.batch_size, num_workers=4
        )
