# Initially based on the example from the mlflow GitHub repository
# https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py

# Incorporating Hydra for configuration following the repo of ymym3412
# https://github.com/ymym3412/Hydra-MLflow-experiment-management

# Following the approach to combining MLflow and Optuna from the repo of StefanieStoppel
# https://github.com/StefanieStoppel/pytorch-mlflow-optuna

# Using data from http://archive.ics.uci.edu/ml/datasets/Wine+Quality

# TODO add in ability to set experiment name and other details for logging

import os
from pathlib import Path
import warnings
import sys
import git

from omegaconf import DictConfig, OmegaConf
import hydra

import optuna

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


def suggest_hyperparameters(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 10, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 1e-4, 10, log=True)
    return alpha, l1_ratio


def objective(trial):

    # Load the data
    train_path = "data/red_wine_train.csv"  # cfg.dataset.train_path
    valid_path = "data/red_wine_valid.csv"  # cfg.dataset.valid_path
    label_column = "quality"  # cfg.dataset.label_column

    cwd = Path.cwd()
    X_train, y_train = load_data(Path(cwd).joinpath(train_path), label_column)
    X_valid, y_valid = load_data(Path(cwd).joinpath(valid_path), label_column)

    # Tell MLflow where to log the experiment
    # mlflow.set_tracking_uri(str(Path(cwd).joinpath("mlruns")))

    with mlflow.start_run():

        # Instantiate the model based on config file
        alpha, l1_ratio = suggest_hyperparameters(trial)

        reg = sklearn.linear_model.ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, random_state=5
        )

        # Fit the model
        reg.fit(X_train, y_train)

        # Predict on the validation set and calculate metrics
        y_pred = reg.predict(X_valid)

        (rmse, mae, r2) = eval_metrics(y_valid, y_pred)

        print(f"Validation set RMSE: {rmse:.2f}")
        print(f"Validation set MAE: {mae:.2f}")
        print(f"Validation set R2: {r2:.2f}")

        # Log all of the hyperparameters to MLflow
        # for key, value in cfg.sklearn_model.items():
        #    if key == "_target_":
        #        pass
        #    else:
        #        mlflow.log_param(key, value)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log the metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log the hydra logs as an MLflow artifact
        mlflow.log_artifact(cwd.joinpath("hydra_output"))

        # Log the model to MLflow
        mlflow.sklearn.log_model(reg, "model")

        return rmse


def main():

    # Only proceed with the experiment if the repository is clean
    repo = git.Repo("~/Documents/CDT/other_learning/mlflow/mlflow-demos")
    assert (
        repo.is_dirty() is False
    ), "Git repository is dirty, please commit before running experiment"

    study = optuna.create_study(
        study_name="wine-quality-elasticnet", direction="minimize"
    )
    study.optimize(objective, n_trials=10)
