# Initially based on the example from the mlflow GitHub repository
# https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py

# Incorporating Hydra for configuration following the repo of ymym3412
# https://github.com/ymym3412/Hydra-MLflow-experiment-management

# Using data from http://archive.ics.uci.edu/ml/datasets/Wine+Quality

import os
from pathlib import Path
import warnings
import sys

from omegaconf import DictConfig, OmegaConf
import hydra

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging


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

    print(OmegaConf.to_yaml(cfg))

    # TODO: Increase flexibility so we can choose an sklearn model and supply
    # different hyperparameters (and, eventually, ranges)
    train_path = cfg.dataset.train_path
    valid_path = cfg.dataset.valid_path
    label_column = cfg.dataset.label_column

    alpha = cfg.sklearn_model.alpha
    l1_ratio = cfg.sklearn_model.l1_ratio
    random_state = cfg.sklearn_model.random_state

    cwd = Path.cwd()
    X_train, y_train = load_data(Path(cwd).joinpath(train_path), label_column)
    X_valid, y_valid = load_data(Path(cwd).joinpath(valid_path), label_column)

    mlflow.set_tracking_uri(str(Path(cwd).joinpath("mlruns")))

    with mlflow.start_run():
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_valid)

        (rmse, mae, r2) = eval_metrics(y_valid, y_pred)

        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"    Validation set RMSE: {rmse:.2f}")
        print(f"    Validation set MAE: {mae:.2f}")
        print(f"    Validation set R2: {r2:.2f}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.log_artifact(cwd.joinpath("hydra_output"))

        mlflow.sklearn.log_model(reg, "model")


if __name__ == "__main__":
    train_eval_model()
