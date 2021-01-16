# Based on the example from the mlflow GitHub repository
# https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py

# Using data from http://archive.ics.uci.edu/ml/datasets/Wine+Quality

import os
from pathlib import Path
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

DATA_PATH = Path("data")
TRAIN_PATH = DATA_PATH.joinpath("red_wine_train.csv")
VALID_PATH = DATA_PATH.joinpath("red_wine_valid.csv")

LABEL_COL = "quality"

SEED = 5

# ElasticNet hyperparameters
alpha = 0.5
l1_ratio = 0.5


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


def train_eval_model():
    X_train, y_train = load_data(TRAIN_PATH, LABEL_COL)
    X_valid, y_valid = load_data(VALID_PATH, LABEL_COL)

    with mlflow.start_run():
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=SEED)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_valid)

        (rmse, mae, r2) = eval_metrics(y_valid, y_pred)

        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}:")
        print(f"    Validation set RMSE: {rmse:.2f}")
        print(f"    Validation set MAE: {mae:.2f}")
        print(f"    Validation set R2: {r2:.2f}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(reg, "model")


if __name__ == "__main__":
    train_eval_model()
