# Initially based on the example from the mlflow GitHub repository
# https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py

# Incorporating Hydra for configuration following the repo of ymym3412
# https://github.com/ymym3412/Hydra-MLflow-mlflow_experiment-management

# Using data from http://archive.ics.uci.edu/ml/datasets/Wine+Quality

import os
from pathlib import Path
import warnings
import sys
import git

from omegaconf import DictConfig, OmegaConf
import hydra

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from jmf_mlflow_utils import get_mlflow_tags

from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data(path, label_col):
    df = pd.read_csv(path, index_col=0)
    X = df.drop(label_col, axis=1)
    y = df[label_col]
    return X, y


def train_eval_model(
    dataset,
    model,
    hyperparameters,
    logdir=None,
    mlflow_client=None,
    trial_mlflow_run_id=None,
    mlflow_parent_experiment_id=None,
    mlflow_parent_run_id=None,
):
    # If creating a child run, create a child run ID
    if mlflow_parent_run_id is not None and mlflow_parent_experiment_id is not None:
        trial_mlflow_run = mlflow_client.create_run(
            experiment_id=mlflow_parent_experiment_id,
            tags=get_mlflow_tags(
                manual_tags={MLFLOW_PARENT_RUN_ID: mlflow_parent_run_id}
            ),
        )
        trial_mlflow_run_id = trial_mlflow_run.info.run_id

    # Load the data
    train_path = dataset.train_path
    valid_path = dataset.valid_path
    label_column = dataset.label_column

    cwd = Path.cwd()
    X_train, y_train = load_data(Path(cwd).joinpath(train_path), label_column)
    X_valid, y_valid = load_data(Path(cwd).joinpath(valid_path), label_column)

    # Set the hyperparameters
    model.set_params(**hyperparameters)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the validation set and calculate metrics
    y_pred = model.predict(X_valid)

    (rmse, mae, r2) = eval_metrics(y_valid, y_pred)

    print(f"Validation set RMSE: {rmse:.2f}")
    print(f"Validation set MAE: {mae:.2f}")
    print(f"Validation set R2: {r2:.2f}")

    # Log all of the hyperparameters to MLflow
    for key, value in hyperparameters.items():
        mlflow_client.log_param(trial_mlflow_run_id, key, value)

    # Log the metrics to MLflow
    mlflow_client.log_metric(trial_mlflow_run_id, "rmse", rmse)
    mlflow_client.log_metric(trial_mlflow_run_id, "r2", r2)
    mlflow_client.log_metric(trial_mlflow_run_id, "mae", mae)

    # If additional logs created, e.g. by Hydra, add as artifact
    if logdir:
        mlflow_client.log_artifact(trial_mlflow_run_id, logdir)

    # Log the model to MLflow
    # mlflow.sklearn.log_model(model, "model")

    return rmse


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # Only proceed with the xperiment if the repository is clean
    if not cfg.debug:
        repo = git.Repo(cfg.repo)
        assert (
            repo.is_dirty() is False
        ), "Git repository is dirty, please commit before running experiment"
    else:
        Path(cfg.debug_output_subdir).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"./{cfg.debug_output_subdir}")

    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    if "azure_mlflow" in cfg.keys():
        ws = Workspace(**cfg.azure_mlflow, auth=AzureCliAuthentication())
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    mlflow_client = MlflowClient()

    try:
        mlflow_experiment_id = mlflow_client.create_experiment(
            cfg.sklearn_train.mlflow_experiment_name
        )
    except:
        mlflow_experiment_id = mlflow_client.get_experiment_by_name(
            cfg.sklearn_train.mlflow_experiment_name
        ).experiment_id

    mlflow_run = mlflow_client.create_run(
        experiment_id=mlflow_experiment_id, tags=get_mlflow_tags()
    )
    mlflow_run_id = mlflow_run.info.run_id

    model = hydra.utils.instantiate(cfg.sklearn_train.model)

    train_eval_model(
        model=model,
        dataset=cfg.dataset,
        hyperparameters=cfg.sklearn_train.hyperparameters,
        logdir=cfg.hydra_logdir,
        mlflow_client=mlflow_client,
        trial_mlflow_run_id=mlflow_run_id,
    )


if __name__ == "__main__":
    main()
