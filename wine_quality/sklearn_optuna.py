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


# TODO Improve this class to handle step/log flags properly
# Possible the best way to do it is by matching keys
class SetHPs:
    def __init__(self, hp_config):
        self.hp_config_float = {}
        self.hp_config_int = {}
        self.hp_config_categorical = {}

        for name in hp_config:
            if hp_config[name]["type"] == "float":
                self.hp_config_float[name] = hp_config[name]
            elif hp_config[name]["type"] == "int":
                self.hp_config_int[name] = hp_config[name]
            elif hp_config[name]["type"] == "categorical":
                self.hp_config_categorical[name] == hp_config[name]
            else:
                raise ValueError("Check hyperparameter search space types")

    def suggest_hyperparameters(self, trial):
        out_dict = {}
        for name in self.hp_config_float.keys():
            out_dict[name] = trial.suggest_float(
                name=name,
                low=self.hp_config_float[name]["low"],
                high=self.hp_config_float[name]["high"],
                # step=self.hp_config_float[name]["step"],
                log=self.hp_config_float[name]["log"],
            )

        return out_dict


# TODO: Think about just loading the data once, and calculating a hash for each file
# that can then be logged in MLflow to confirm using same file each time.
class Objective:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hp = SetHPs(cfg.sklearn_tune.search_ranges)

    def __call__(self, trial):

        # Load the data
        train_path = self.cfg.dataset.train_path
        valid_path = self.cfg.dataset.valid_path
        label_column = self.cfg.dataset.label_column

        cwd = Path.cwd()
        X_train, y_train = load_data(Path(cwd).joinpath(train_path), label_column)
        X_valid, y_valid = load_data(Path(cwd).joinpath(valid_path), label_column)

        # Tell MLflow where to log the experiment
        # mlflow.set_tracking_uri(str(Path(cwd).joinpath("mlruns")))

        with mlflow.start_run():

            # Instantiate the model based on config file
            current_hps = self.hp.suggest_hyperparameters(trial)
            print(current_hps)

            reg = hydra.utils.instantiate(self.cfg.sklearn_tune.model)
            reg.set_params(**current_hps)

            # Fit the model
            reg.fit(X_train, y_train)

            # Predict on the validation set and calculate metrics
            y_pred = reg.predict(X_valid)

            (rmse, mae, r2) = eval_metrics(y_valid, y_pred)

            print(f"Validation set RMSE: {rmse:.2f}")
            print(f"Validation set MAE: {mae:.2f}")
            print(f"Validation set R2: {r2:.2f}")

            # Log all of the hyperparameters to MLflow
            for key, value in current_hps.items():
                mlflow.log_param(key, value)

            # Log the metrics to MLflow
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Log the hydra logs as an MLflow artifact
            mlflow.log_artifact(cwd.joinpath("hydra_output"))

            # Log the model to MLflow
            mlflow.sklearn.log_model(reg, "model")

            return rmse


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    # Only proceed with the experiment if the repository is clean
    repo = git.Repo("~/Documents/CDT/other_learning/mlflow/mlflow-demos")
    assert (
        repo.is_dirty() is False
    ), "Git repository is dirty, please commit before running experiment"

    study = optuna.create_study(
        study_name="wine-quality-elasticnet", direction="minimize"
    )
    study.optimize(Objective(cfg), n_trials=10)


if __name__ == "__main__":
    main()