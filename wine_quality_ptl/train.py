from datamodule import WineDataModule
from network import LitWine
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import os
import shutil

mlflow.pytorch.autolog(log_models=False)

wine_dm = WineDataModule(batch_size=8)
model = LitWine()
trainer = Trainer(max_epochs=20)


with mlflow.start_run() as run:
    trainer.fit(model, wine_dm)

    cp_name = os.listdir("lightning_logs/version_0/checkpoints")[0]
    best_checkpoint = f"lightning_logs/version_0/checkpoints/{cp_name}"
    best_model = LitWine.load_from_checkpoint(best_checkpoint)
    mlflow.pytorch.log_model(best_model, "best_model")

    mlflow.log_artifact("lightning_logs/")
    shutil.rmtree("lightning_logs")
