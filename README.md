# mlflow-demos
Explore setting up simple experiment tracking workflows using MLflow, uisng Hydra for configuration and Optuna for hyperparameter tuning.

The script `sklearn_optuna.py` can take an optional configuration file for a RDB backend for Optuna (e.g. `conf/optuna_storage/example.yaml`). The construction of the RDB URL is set up for a PostgreSQL database hosted on Azure. If the RDB backend is used multiple copies of the script can be run to peform distributed tuning. The next step is to also include remote storage for MLFlow.

MLFLow: https://www.mlflow.org/

Hydra: https://hydra.cc/

Optuna: https://optuna.readthedocs.io

The Azure free trial currently includes a PostgreSQL database that can be used as remote storage for Optuna. This can be set up by following the quick start guide: https://docs.microsoft.com/en-us/azure/postgresql/quickstart-create-server-database-portal. 
