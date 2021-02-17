import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam


class LitWine(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(11, 128)
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = F.mse_loss(y_pred, y)
        self.log("valid_mse", val_loss)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-5)
