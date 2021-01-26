from datamodule import WineDataModule
from network import LitWine
from pytorch_lightning.trainer import Trainer


wine_dm = WineDataModule(batch_size=64)
model = LitWine()
trainer = Trainer(max_epochs=20)
trainer.fit(model, wine_dm)
