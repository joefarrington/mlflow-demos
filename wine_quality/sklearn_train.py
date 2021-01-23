# Initially based on the example from the mlflow GitHub repository
# https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py

# Incorporating Hydra for configuration following the repo of ymym3412
# https://github.com/ymym3412/Hydra-MLflow-experiment-management

# Using data from http://archive.ics.uci.edu/ml/datasets/Wine+Quality

# TODO add in ability to set experiment name and other details for logging

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


@hydra.main(config_path="conf", config_name="config")
def train_eval_model(cfg):

    # Only proceed with the experiment if the repository is clean
    repo = git.Repo(cfg.repo)
    assert (
        repo.is_dirty() is False
    ), "Git repository is dirty, please commit before running experiment"

    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    # Load the data
    train_path = cfg.dataset.train_path
    valid_path = cfg.dataset.valid_path
    label_column = cfg.dataset.label_column

    cwd = Path.cwd()
    X_train, y_train = load_data(Path(cwd).joinpath(train_path), label_column)
    X_valid, y_valid = load_data(Path(cwd).joinpath(valid_path), label_column)

    # Tell MLflow where to log the experiment
    mlflow.set_tracking_uri(str(Path(cwd).joinpath("mlruns")))

    with mlflow.start_run():

        # Instantiate the model based on config file
        reg = hydra.utils.instantiate(cfg.sklearn_model)

        # Fit the model
        reg.fit(X_train, y_train)

        # Predict on the validation set and calculate metrics
        y_pred = reg.predict(X_valid)

        (rmse, mae, r2) = eval_metrics(y_valid, y_pred)

        print(f"Validation set RMSE: {rmse:.2f}")
        print(f"Validation set MAE: {mae:.2f}")
        print(f"Validation set R2: {r2:.2f}")

        # Log all of the hyperparameters to MLflow
        for key, value in cfg.sklearn_model.items():
            if key == "_target_":
                pass
            else:
                mlflow.log_param(key, value)

        # Log the metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log the hydra logs as an MLflow artifact
        temp_hydra_log_path = cwd.joinpath(cfg.hydra_logdir)
        mlflow.log_artifact(temp_hydra_log_path)

        # Log the model to MLflow
        mlflow.sklearn.log_model(reg, "model")


if __name__ == "__main__":
    train_eval_model()
