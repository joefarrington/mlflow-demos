import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
import pandas as pd

class WineDataset(Dataset):
    def __init__(self, path, label_col):
        df = pd.read_csv(path, index_col=0)
        X = df.drop(label_col, axis=1).values
        y = df[label_col].values.reshape(-1, 1)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WineDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train = WineDataset("data/red_wine_train.csv", "quality")
        self.valid = WineDataset("data/red_wine_valid.csv", "quality")

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, self.batch_size)
