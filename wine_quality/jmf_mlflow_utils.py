from mlflow.tracking import MlflowClient
import mlflow.projects.utils as mlflow_utils
from mlflow.projects.utils import *
from mlflow.tracking.context.registry import resolve_tags


def get_mlflow_tags(filename=None, manual_tags=None):
    # Can specify filename as string
    # for example for Jupyter where os.path.basename(__file__)
    # does not work

    # Use specified filename if provided
    # Otherwise resolve automatically
    if filename:
        source_name = filename
    else:
        source_name = resolve_tags()["mlflow.source.name"]

    # Use specified working directory if provided

    work_dir = os.getcwd()

    source_version = mlflow_utils._get_git_commit(work_dir)
    tags = {
        MLFLOW_USER: mlflow_utils._get_user(),
        MLFLOW_SOURCE_NAME: source_name,
    }
    if source_version is not None:
        tags[MLFLOW_GIT_COMMIT] = source_version

    repo_url = mlflow_utils._get_git_repo_url(work_dir)
    if repo_url is not None:
        tags[MLFLOW_GIT_REPO_URL] = repo_url
        tags[LEGACY_MLFLOW_GIT_REPO_URL] = repo_url

    if manual_tags:
        for k, v in manual_tags.items():
            tags[k] = v

    return tags
