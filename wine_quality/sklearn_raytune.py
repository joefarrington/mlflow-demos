import os
from pathlib import Path
import warnings
import sys
import git

from omegaconf import DictConfig, OmegaConf
import hydra

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from ray import tune
from ray.tune.integration.mlflow import mlflow_mixin


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data(path, label_col):
    df = pd.read_csv(path, index_col=0)
    X = df.drop(label_col, axis=1)
    y = df[[label_col]]
    return X, y


@mlflow_mixin
def train_fn(config):
    reg = sklearn.linear_model.ElasticNet(
        alpha=self.config["alpha"], l1_ratio=self.config["l1_ratio"], random_state=5
    )

    reg.fit(self.config["X_train"], self.config["y_train"])

    y_pred = reg.predict(self.config["X_valid"])

    # Predict on the validation set and calculate metrics
    y_pred = reg.predict(self.config["X_valid"])

    (rmse, mae, r2) = eval_metrics(self.config["y_valid"], y_pred)

    # Log all of the hyperparameters to MLflow
    mlflow.log_param("alpha", self.config["alpha"])
    mlflow.log_param("l1_ratio", self.config["l1_ratio"])

    # Log the metrics to MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    # Log the model to MLflow
    mlflow.sklearn.log_model(reg, "model")

    # Use the RMSE to select next values
    tune.report(loss=rmse, done=True)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    # Load the data
    train_path = cfg.dataset.train_path
    valid_path = cfg.dataset.valid_path
    label_column = cfg.dataset.label_column

    cwd = Path.cwd()
    X_train, y_train = load_data(Path(cwd).joinpath(train_path), label_column)
    X_valid, y_valid = load_data(Path(cwd).joinpath(valid_path), label_column)

    mlflow.create_experiment("wine_quality_raytune")

    config = {
        "alpha": tune.choice([0.1, 0.5, 1]),
        "l1_ratio": tune.choice([0.1, 0.5, 1]),
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "mlflow": {
            "experiment_name": "wine_quality_raytune",
            "tracking_uri": mlflow.get_tracking_uri(),
        },
    }

    tune.run(train_fn, config)
