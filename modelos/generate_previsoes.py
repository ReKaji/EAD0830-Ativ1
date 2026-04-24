from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

# Import helpers from existing modules
from modelos.p3_modelos_classicos import (
    resolve_project_root,
    EXPECTED_COLUMNS,
    TRAIN_SIZE,
    HORIZON,
    seasonal_naive_forecast,
    select_best_holt_winters,
    fit_holt_winters,
    fit_sarima_grid_search,
)
from modelos.p5_prophet import create_prophet_dataframe, train_prophet, forecast_prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX


MODEL_MAP = {
    "#1": "seasonal_naive",
    "#2": "holt_winters",
    "#3": "sarima",
    "#4": "sarima",
    "#5": "sarima",
    "#6": "sarima",
    "#7": "holt_winters",
    "#8": "sarima",
    "#9": "sarima",
    "#10": "holt_winters",
    "#11": "prophet",
}


def forecast_series(series: pd.Series, model_name: str, horizon: int = 12, seasonal_periods: int = 12) -> np.ndarray:
    y = series.to_numpy(dtype=float)
    full = y.copy()

    if model_name == "seasonal_naive":
        return seasonal_naive_forecast(full, horizon, seasonal_periods)

    # For model selection we follow the same train/valid split used in the exercises
    train = y[:TRAIN_SIZE]
    valid = y[TRAIN_SIZE : TRAIN_SIZE + HORIZON]

    if model_name == "holt_winters":
        best = select_best_holt_winters(train, valid, seasonal_periods)
        if best is None:
            return seasonal_naive_forecast(full, horizon, seasonal_periods)
        hw_config = best["config"] if isinstance(best, dict) and "config" in best else best
        pred, _ = fit_holt_winters(full, horizon, hw_config, seasonal_periods)
        return np.asarray(pred, dtype=float)

    if model_name == "sarima":
        best = fit_sarima_grid_search(train, valid)
        if best is None:
            return seasonal_naive_forecast(full, horizon, seasonal_periods)
        cfg = best["config"] if isinstance(best, dict) and "config" in best else best
        order = tuple(cfg["order"]) if "order" in cfg else tuple(cfg["order"])
        seasonal_order = tuple(cfg["seasonal_order"]) if "seasonal_order" in cfg else tuple(cfg["seasonal_order"])
        model = SARIMAX(
            full,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=50)
        pred = np.asarray(fitted.forecast(horizon), dtype=float)
        return pred

    if model_name == "prophet":
        df_prop = create_prophet_dataframe(series)
        model = train_prophet(df_prop)
        forecast_df = forecast_prophet(model, periods=horizon)
        preds = forecast_df.iloc[-horizon:]["yhat"].values
        return np.asarray(preds, dtype=float)

    raise ValueError(f"Modelo desconhecido: {model_name}")


def main() -> None:
    project_root = resolve_project_root()
    data_path = project_root / "data" / "DadosCompeticao.xlsx"
    out_dir = project_root / "resultados" / "previsoes_12_meses"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path, sheet_name=0).copy()

    # Validate columns
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes na base: {missing}")

    final_forecasts: dict[str, np.ndarray] = {}
    for col in EXPECTED_COLUMNS:
        model_name = MODEL_MAP.get(col, "seasonal_naive")
        print(f"Gerando previsão para {col} usando {model_name}")
        preds = forecast_series(df[col], model_name, horizon=12, seasonal_periods=12)
        final_forecasts[col] = preds

    final_df = pd.DataFrame(final_forecasts)

    out_path = out_dir / "Previsoes_12_meses.xlsx"
    final_df.to_excel(out_path, index=False)

    # Also write a small metadata json
    meta = {
        "input": str(data_path),
        "output": str(out_path),
        "models": MODEL_MAP,
        "horizon": 12,
    }
    (out_dir / "previsoes_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Previsões salvas em: {out_path}")


if __name__ == "__main__":
    main()
