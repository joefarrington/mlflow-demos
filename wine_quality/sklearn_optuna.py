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

from sklearn_train import train_eval_model


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


class Objective:
    def __init__(self, cfg, experiment_id):
        self.cfg = cfg
        self.hp = SetHPs(cfg.sklearn_tune.hyperparameters)
        self.experiment_id = experiment_id

    def __call__(self, trial):

        hyperparameters = self.hp.suggest_hyperparameters(trial)
        model = hydra.utils.instantiate(self.cfg.sklearn_tune.model)
        print(hyperparameters)

        val_loss = train_eval_model(
            dataset=self.cfg.dataset,
            model=model,
            hyperparameters=hyperparameters,
            logdir=self.cfg.hydra_logdir,
            experiment_id=self.experiment_id,
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

    try:
        mlflow.create_experiment(cfg.sklearn_tune.experiment_id)
    except:
        pass

    experiment_id = mlflow.get_experiment_by_name(
        cfg.sklearn_tune.experiment_id
    ).experiment_id

    # TODO: Consider saving the Study object and/or creating Optuna visualizations

    sampler = TPESampler(seed=cfg.sklearn_tune.hyperparameters.fixed.random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(Objective(cfg, experiment_id), n_trials=cfg.sklearn_tune.n_trials)


if __name__ == "__main__":
    main()