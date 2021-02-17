from datamodule import WineDataModule
from network import LitWine
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from time import strftime, localtime
import os

mlflow.pytorch.autolog(log_models=False)

wine_dm = WineDataModule(batch_size=8)
model = LitWine()
lightning_log_path = f"./lightning_logs/{strftime('%Y-%m-%d', localtime())}/{strftime('%H-%M-%S', localtime())}"
trainer = Trainer(max_epochs=20, default_root_dir=lightning_log_path)


with mlflow.start_run() as run:
    trainer.fit(model, wine_dm)
    cp_name = os.listdir(lightning_log_path + "/lightning_logs/version_0/checkpoints")[
        0
    ]
    best_checkpoint = (
        lightning_log_path + f"/lightning_logs/version_0/checkpoints/{cp_name}"
    )

    best_model = LitWine.load_from_checkpoint(best_checkpoint)
    mlflow.pytorch.log_model(best_model, "best_model")

    mlflow.log_artifact(lightning_log_path + "/lightning_logs/")
