def suggest_hyperparameters(trial):
    alpha = trial.suggest_float("alpha", 1e-4, 1, log=True)
    l1_ratio = trial.suggest_float("l1_ratio", 1e-4, 1, log=True)
    return alpha, l1_ratio