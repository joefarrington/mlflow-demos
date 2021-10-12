# mlflow-demos

Explore setting up simple experiment tracking workflows using MLflow, uisng Hydra for configuration and Optuna for hyperparameter tuning.

The script `sklearn_optuna.py` can take an optional configuration file for a RDB backend for Optuna (e.g. `conf/optuna_storage/example.yaml`). The construction of the RDB URL is set up for a PostgreSQL database hosted on Azure. If the RDB backend is used multiple copies of the script can be run to peform distributed tuning.

Additionally, both `sklearn_train.py` and `sklearn_optuna.py` can taken an optional configuration file to uze Azure to track MLflow experiments (e.g. `conf/azure_mlflow/example.yaml`). The code currently assumes that authorization for access to the Azure workspace will be performed using the Azure CLI (see: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli and https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb for more information).

MLfLow: https://www.mlflow.org/

Hydra: https://hydra.cc/

Optuna: https://optuna.readthedocs.io

I followed Simon Hessner's guide to using MLflow child runs for each trial when tuning using Optuna: https://simonhessner.de/mlflow-optuna-parallel-hyper-parameter-optimization-and-logging/

The Azure free trial currently includes a PostgreSQL database that can be used as remote storage for Optuna. This can be set up by following the quick start guide: https://docs.microsoft.com/en-us/azure/postgresql/quickstart-create-server-database-portal.
