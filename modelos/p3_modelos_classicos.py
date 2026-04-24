from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

TRAIN_SIZE = 252
HORIZON = 12
SEASONAL_PERIODS = 12
EXPECTED_COLUMNS = [f"#{i}" for i in range(1, 12)]


@dataclass
class Config:
    input_path: Path
    output_dir: Path
    train_size: int = TRAIN_SIZE
    horizon: int = HORIZON
    seasonal_periods: int = SEASONAL_PERIODS
    sheet_name: str | int = 0


def resolve_project_root() -> Path:
    """
    Resolve a raiz do repositório assumindo que este arquivo vive em:
    <repo>/modelos/p3_modelos_classicos.py
    """
    return Path(__file__).resolve().parents[1]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_series_table(config: Config) -> pd.DataFrame:
    path = config.input_path
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {path}\n"
            "Confirme se a base está em data/DadosCompeticao.xlsx."
        )

    df = pd.read_excel(path, sheet_name=config.sheet_name).copy()

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "A planilha precisa conter exatamente as colunas #1 até #11.\n"
            f"Colunas ausentes: {missing}\n"
            f"Colunas encontradas: {list(df.columns)}"
        )

    df = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        bad_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(
            "A base contém valores ausentes ou não numéricos após a conversão.\n"
            f"Revise as colunas: {bad_cols}"
        )

    expected_rows = config.train_size + config.horizon
    if len(df) != expected_rows:
        raise ValueError(
            f"A base deve ter exatamente {expected_rows} linhas "
            f"(treino={config.train_size}, teste={config.horizon}). "
            f"Foram encontradas {len(df)} linhas."
        )

    return df.reset_index(drop=True)


def seasonal_naive_forecast(train: np.ndarray, horizon: int, seasonal_periods: int) -> np.ndarray:
    """
    Baseline sazonal: repete os últimos 12 meses.
    """
    if len(train) < seasonal_periods:
        return np.repeat(train[-1], horizon)

    last_season = np.asarray(train[-seasonal_periods:], dtype=float)
    reps = int(np.ceil(horizon / seasonal_periods))
    return np.tile(last_season, reps)[:horizon]


def build_hw_configs(series: np.ndarray) -> list[dict[str, Any]]:
    """
    Testa algumas configurações simples e robustas de Holt-Winters.
    Para modelos multiplicativos, todos os valores precisam ser > 0.
    """
    series = np.asarray(series, dtype=float)
    positive_only = np.all(series > 0)

    configs: list[dict[str, Any]] = [
        {"trend": None, "seasonal": "add", "damped_trend": False},
        {"trend": "add", "seasonal": "add", "damped_trend": False},
        {"trend": "add", "seasonal": "add", "damped_trend": True},
    ]

    if positive_only:
        configs.extend(
            [
                {"trend": None, "seasonal": "mul", "damped_trend": False},
                {"trend": "add", "seasonal": "mul", "damped_trend": False},
                {"trend": "add", "seasonal": "mul", "damped_trend": True},
            ]
        )

    return configs


def fit_holt_winters(
    train: np.ndarray,
    horizon: int,
    hw_config: dict[str, Any],
    seasonal_periods: int,
) -> tuple[np.ndarray, Any]:
    model = ExponentialSmoothing(
        train,
        trend=hw_config["trend"],
        seasonal=hw_config["seasonal"],
        seasonal_periods=seasonal_periods,
        damped_trend=hw_config["damped_trend"],
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, use_brute=False)
    forecast = fitted.forecast(horizon)
    return np.asarray(forecast, dtype=float), fitted


def select_best_holt_winters(train: np.ndarray, valid: np.ndarray, seasonal_periods: int) -> dict[str, Any] | None:
    best_result: dict[str, Any] | None = None

    for hw_config in build_hw_configs(train):
        try:
            pred, fitted = fit_holt_winters(train, len(valid), hw_config, seasonal_periods)
            result = {
                "model": "holt_winters",
                "rmse": rmse(valid, pred),
                "config": hw_config,
                "pred_valid": pred,
                "fitted": fitted,
            }
            if best_result is None or result["rmse"] < best_result["rmse"]:
                best_result = result
        except Exception:
            continue

    return best_result


def build_sarima_candidates() -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """
    Busca restrita e estável para SARIMA mensal.
    Mantém o tempo de execução baixo para o grupo conseguir rodar em qualquer ambiente.
    """
    return [
        ((1, 1, 1), (0, 1, 0, 12)),
        ((0, 1, 1), (0, 1, 1, 12)),
        ((1, 1, 0), (1, 1, 0, 12)),
    ]


def fit_sarima_grid_search(train: np.ndarray, valid: np.ndarray) -> dict[str, Any] | None:
    best_result: dict[str, Any] | None = None

    for order, seasonal_order in build_sarima_candidates():
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=50)
            pred = np.asarray(fitted.forecast(len(valid)), dtype=float)

            result = {
                "model": "sarima",
                "rmse": rmse(valid, pred),
                "config": {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "aic": float(fitted.aic),
                },
                "pred_valid": pred,
                "fitted": fitted,
            }

            if best_result is None or result["rmse"] < best_result["rmse"]:
                best_result = result
        except Exception:
            continue

    return best_result


def evaluate_one_series(series_name: str, values: pd.Series, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = values.to_numpy(dtype=float)
    train = y[: config.train_size]
    valid = y[config.train_size : config.train_size + config.horizon]

    model_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []

    # 1) Seasonal Naive
    pred_naive = seasonal_naive_forecast(train, config.horizon, config.seasonal_periods)
    model_rows.append(
        {
            "serie": series_name,
            "modelo": "seasonal_naive",
            "rmse": rmse(valid, pred_naive),
            "config": "",
        }
    )
    for step, (actual, pred) in enumerate(zip(valid, pred_naive), start=1):
        holdout_rows.append(
            {
                "serie": series_name,
                "modelo": "seasonal_naive",
                "h": step,
                "real": float(actual),
                "previsto": float(pred),
                "erro": float(actual - pred),
            }
        )

    # 2) Holt-Winters
    best_hw = select_best_holt_winters(train, valid, config.seasonal_periods)
    if best_hw is not None:
        model_rows.append(
            {
                "serie": series_name,
                "modelo": "holt_winters",
                "rmse": best_hw["rmse"],
                "config": json.dumps(best_hw["config"], ensure_ascii=False),
            }
        )
        for step, (actual, pred) in enumerate(zip(valid, best_hw["pred_valid"]), start=1):
            holdout_rows.append(
                {
                    "serie": series_name,
                    "modelo": "holt_winters",
                    "h": step,
                    "real": float(actual),
                    "previsto": float(pred),
                    "erro": float(actual - pred),
                }
            )

    # 3) SARIMA
    best_sarima = fit_sarima_grid_search(train, valid)
    if best_sarima is not None:
        model_rows.append(
            {
                "serie": series_name,
                "modelo": "sarima",
                "rmse": best_sarima["rmse"],
                "config": json.dumps(best_sarima["config"], ensure_ascii=False),
            }
        )
        for step, (actual, pred) in enumerate(zip(valid, best_sarima["pred_valid"]), start=1):
            holdout_rows.append(
                {
                    "serie": series_name,
                    "modelo": "sarima",
                    "h": step,
                    "real": float(actual),
                    "previsto": float(pred),
                    "erro": float(actual - pred),
                }
            )
    else:
        model_rows.append(
            {
                "serie": series_name,
                "modelo": "sarima",
                "rmse": np.nan,
                "config": "nenhum candidato convergiu",
            }
        )

    performance_df = (
        pd.DataFrame(model_rows)
        .sort_values(["serie", "rmse", "modelo"], na_position="last")
        .reset_index(drop=True)
    )
    holdout_df = pd.DataFrame(holdout_rows)
    return performance_df, holdout_df


def refit_best_and_forecast(values: pd.Series, best_row: pd.Series, config: Config) -> np.ndarray:
    y = values.to_numpy(dtype=float)
    full = y.copy()
    model_name = best_row["modelo"]
    model_config = best_row["config"]

    if model_name == "seasonal_naive":
        return seasonal_naive_forecast(full, config.horizon, config.seasonal_periods)

    if model_name == "holt_winters":
        hw_config = json.loads(model_config)
        pred, _ = fit_holt_winters(full, config.horizon, hw_config, config.seasonal_periods)
        return pred

    if model_name == "sarima":
        cfg = json.loads(model_config)
        model = SARIMAX(
            full,
            order=tuple(cfg["order"]),
            seasonal_order=tuple(cfg["seasonal_order"]),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=50)
        pred = np.asarray(fitted.forecast(config.horizon), dtype=float)
        return pred

    raise ValueError(f"Modelo desconhecido para refit: {model_name}")


def save_outputs(
    performance_df: pd.DataFrame,
    best_models_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    final_forecast_df: pd.DataFrame,
    config: Config,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    performance_path = config.output_dir / "p3_performance_modelos.xlsx"
    best_models_path = config.output_dir / "p3_melhor_modelo_por_serie.xlsx"
    holdout_path = config.output_dir / "p3_previsoes_holdout.xlsx"
    final_forecast_path = config.output_dir / "p3_previsoes_finais.xlsx"
    summary_path = config.output_dir / "p3_resumo_execucao.json"

    performance_df.to_excel(performance_path, index=False)
    best_models_df.to_excel(best_models_path, index=False)
    holdout_df.to_excel(holdout_path, index=False)
    final_forecast_df.to_excel(final_forecast_path, index=False)

    summary = {
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "train_size": config.train_size,
        "horizon": config.horizon,
        "seasonal_periods": config.seasonal_periods,
        "series": EXPECTED_COLUMNS,
        "arquivos_gerados": [
            performance_path.name,
            best_models_path.name,
            holdout_path.name,
            final_forecast_path.name,
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(config: Config) -> None:
    df = load_series_table(config)

    all_performance = []
    all_holdout = []

    for series_name in EXPECTED_COLUMNS:
        performance_df, holdout_df = evaluate_one_series(series_name, df[series_name], config)
        all_performance.append(performance_df)
        all_holdout.append(holdout_df)

    performance_df = pd.concat(all_performance, ignore_index=True)
    performance_df = performance_df.sort_values(["serie", "rmse", "modelo"], na_position="last").reset_index(drop=True)

    best_models_df = (
        performance_df.dropna(subset=["rmse"])
        .sort_values(["serie", "rmse", "modelo"])
        .groupby("serie", as_index=False)
        .first()
        .rename(columns={"rmse": "rmse_holdout"})
    )

    holdout_df = pd.concat(all_holdout, ignore_index=True)

    final_forecasts = {}
    for series_name in EXPECTED_COLUMNS:
        best_row = best_models_df.loc[best_models_df["serie"] == series_name].iloc[0]
        final_forecasts[series_name] = refit_best_and_forecast(df[series_name], best_row, config)

    final_forecast_df = pd.DataFrame(final_forecasts)
    final_forecast_df.insert(0, "h", np.arange(1, config.horizon + 1))

    save_outputs(performance_df, best_models_df, holdout_df, final_forecast_df, config)

    print("Execução concluída com sucesso.")
    print(f"Base lida de: {config.input_path}")
    print(f"Resultados salvos em: {config.output_dir}\n")
    print("Melhor modelo por série:")
    print(best_models_df.to_string(index=False))


def main() -> None:
    project_root = resolve_project_root()
    config = Config(
        input_path=project_root / "data" / "DadosCompeticao.xlsx",
        output_dir=project_root / "resultados" / "p3_modelos_classicos",
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
