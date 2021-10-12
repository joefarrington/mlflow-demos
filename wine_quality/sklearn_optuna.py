# Initially based on the example from the mlflow GitHub repository
# https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py

# Incorporating Hydra for configuration following the repo of ymym3412
# https://github.com/ymym3412/Hydra-MLflow-experiment-management

# Following the approach to combining MLflow and Optuna from the repo of StefanieStoppel
# https://github.com/StefanieStoppel/pytorch-mlflow-optuna

# Using data from http://archive.ics.uci.edu/ml/datasets/Wine+Quality

import os
from pathlib import Path
import warnings
import sys
import git
import shutil

from omegaconf import DictConfig, OmegaConf
import hydra

import optuna
from optuna.samplers import TPESampler

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from jmf_mlflow_utils import get_mlflow_tags

from sklearn_train import train_eval_model

from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication


class SetHPs:
    def __init__(self, hp_config):
        self.hp_config_fixed = OmegaConf.to_container(hp_config.fixed)
        self.hp_config_search = OmegaConf.to_container(hp_config.search_ranges)

        self.hp_config_float = {}
        self.hp_config_int = {}
        self.hp_config_categorical = {}

        for name in self.hp_config_search:
            if self.hp_config_search[name]["type"] == "float":
                self.hp_config_float[name] = self.hp_config_search[name]
                del self.hp_config_float[name]["type"]
            elif self.hp_config_search[name]["type"] == "int":
                self.hp_config_int[name] = self.hp_config_search[name]
                del self.hp_config_int[name]["type"]
            elif self.hp_config_search[name]["type"] == "categorical":
                self.hp_config_categorical[name] = self.hp_config_search[name]
                del self.hp_config_categorical[name]["type"]
            else:
                raise ValueError("Check hyperparameter search space types")

    def suggest_hyperparameters(self, trial):
        out_dict = self.hp_config_fixed.copy()

        for name in self.hp_config_float.keys():
            out_dict[name] = trial.suggest_float(
                name=name, **self.hp_config_float[name]
            )
        for name in self.hp_config_int.keys():
            out_dict[name] = trial.suggest_int(name=name, **self.hp_config_int[name])
        for name in self.hp_config_categorical.keys():
            out_dict[name] = trial.suggest_categorical(
                name=name, **self.hp_config_categorical[name]
            )

        return out_dict


def construct_azure_postgres_url(db_user, db_pass, db_host, db_port, db_name):
    storage_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}?sslmode=require"
    return storage_url


class Objective:
    def __init__(
        self, cfg, mlflow_client, mlflow_parent_experiment_id, mlflow_parent_run_id
    ):
        self.cfg = cfg
        self.hp = SetHPs(cfg.sklearn_tune.hyperparameters)
        self.mlflow_client = mlflow_client
        self.mlflow_parent_experiment_id = mlflow_parent_experiment_id
        self.mlflow_parent_run_id = mlflow_parent_run_id

    def __call__(self, trial):

        hyperparameters = self.hp.suggest_hyperparameters(trial)
        model = hydra.utils.instantiate(self.cfg.sklearn_tune.model)
        print(hyperparameters)

        val_loss = train_eval_model(
            dataset=self.cfg.dataset,
            model=model,
            hyperparameters=hyperparameters,
            logdir=self.cfg.hydra_logdir,
            mlflow_client=self.mlflow_client,
            mlflow_parent_experiment_id=self.mlflow_parent_experiment_id,
            mlflow_parent_run_id=self.mlflow_parent_run_id,
        )

        return val_loss


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    # Only proceed with the experiment if the repository is clean
    if not cfg.debug:
        repo = git.Repo(cfg.repo)
        assert (
            repo.is_dirty() is False
        ), "Git repository is dirty, please commit before running experiment"
    else:
        Path(cfg.debug_output_subdir).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"./{cfg.debug_output_subdir}")

    if "azure_mlflow" in cfg.keys():
        ws = Workspace(**cfg.azure_mlflow, auth=AzureCliAuthentication())
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    mlflow_client = MlflowClient()

    try:
        mlflow_parent_experiment_id = mlflow_client.create_experiment(
            cfg.sklearn_tune.mlflow_experiment_name
        )
    except:
        mlflow_parent_experiment_id = mlflow_client.get_experiment_by_name(
            cfg.sklearn_tune.mlflow_experiment_name
        ).experiment_id

    mlflow_parent_run = mlflow_client.create_run(
        experiment_id=mlflow_parent_experiment_id, tags=get_mlflow_tags()
    )
    mlflow_parent_run_id = mlflow_parent_run.info.run_id

    mlflow_client.log_param(mlflow_parent_run_id, "n_trials", cfg.sklearn_tune.n_trials)

    # TODO: Consider creating Optuna visualizations

    sampler = TPESampler(seed=cfg.sklearn_tune.hyperparameters.fixed.random_state)

    # Check is we've specified remote storage
    if "optuna_storage" in cfg.keys():
        storage = construct_azure_postgres_url(**cfg["optuna_storage"])
    else:
        storage = None

    # If using storage and the study already exists, then use the exisiting study
    # This allows us to do further trials on an existing study, or do distributed optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=cfg.sklearn_tune.optuna_study_name,
        load_if_exists=True,
    )

    study.optimize(
        Objective(
            cfg, mlflow_client, mlflow_parent_experiment_id, mlflow_parent_run_id
        ),
        n_trials=cfg.sklearn_tune.n_trials,
    )

    mlflow_client.log_metric(mlflow_parent_run_id, "best_trial_rmse", study.best_value)
    for k, v in study.best_params.items():
        mlflow_client.log_param(mlflow_parent_run_id, f"best_trial_{k}", v)


if __name__ == "__main__":
    main()
