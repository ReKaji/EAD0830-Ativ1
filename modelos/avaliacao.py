import pandas as pd
import numpy as np


def rmse(y_true, y_pred) -> float:
    """Calcula RMSE entre `y_true` e `y_pred`.
    Aceita listas, arrays ou Series.
    """
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo comprimento")
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def wrmse(y_true, y_pred) -> float:
    y_true = pd.DataFrame(y_true).astype(float)
    y_pred = pd.DataFrame(y_pred).astype(float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true e y_pred devem ter a mesma forma")

    # RMSE por série
    rmse_per_col = ((y_true - y_pred) ** 2).mean(axis=0).pow(0.5).values

    # média simples
    return float(np.mean(rmse_per_col))