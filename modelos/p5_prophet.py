import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from pathlib import Path


def create_prophet_dataframe(series: pd.Series):
    """Converte a série em DataFrame com colunas `ds` e `y` para Prophet."""
    dates = pd.date_range(start="2000-01-01", periods=len(series), freq="MS")
    return pd.DataFrame({"ds": dates, "y": series.values})


def train_test_split_time_series(df: pd.DataFrame, test_size: int = 36):
    """Divide mantendo a ordem temporal."""
    train = df.iloc[:-test_size].reset_index(drop=True)
    test = df.iloc[-test_size:].reset_index(drop=True)
    return train, test


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def train_prophet(train_df: pd.DataFrame):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95,
    )
    model.fit(train_df)
    return model


def forecast_prophet(model: Prophet, periods: int, freq: str = "MS"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


def evaluate_series(series: pd.Series, test_size: int = 36):
    df_prophet = create_prophet_dataframe(series)
    train_df, test_df = train_test_split_time_series(df_prophet, test_size=test_size)
    model = train_prophet(train_df)

    horizon = len(test_df)
    forecast_df = forecast_prophet(model, periods=horizon)
    forecast_test = forecast_df.iloc[-horizon:]["yhat"].values

    error = rmse(test_df["y"].values, forecast_test)
    return error, model


def run_prophet(df: pd.DataFrame, test_size: int = 36):
    results = []
    for column in df.columns:
        error, model = evaluate_series(df[column], test_size)
        results.append({"series": column, "rmse": error, "model": model})
    return pd.DataFrame(results)


def weighted_rmse(results_df: pd.DataFrame):
    rmses = results_df["rmse"].values
    weights = 1.0 / rmses
    weights /= weights.sum()
    return float(np.dot(weights, rmses))


def plot_forecasts(df: pd.DataFrame, results_df: pd.DataFrame, horizon: int = 12):
    n_series = len(df.columns)
    ncols = 4
    nrows = int(np.ceil(n_series / ncols))
    wrmse = weighted_rmse(results_df)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    fig.suptitle(
        f"Prophet — Previsões {horizon} meses (WRMSE = {wrmse:.4f})",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    axes_flat = axes.flatten()

    for idx, row in results_df.iterrows():
        ax = axes_flat[idx]
        series = df[row["series"]]
        model = row["model"]
        rmse_val = row["rmse"]

        full_df = create_prophet_dataframe(series)
        forecast_df = forecast_prophet(model, periods=horizon)
        future_preds = forecast_df.iloc[-horizon:]["yhat"].values

        n_hist = len(series)
        x_hist = np.arange(n_hist)
        x_future = np.arange(n_hist - 1, n_hist + horizon)
        future_with_anchor = np.concatenate([[series.values[-1]], future_preds])

        ax.plot(x_hist, series.values, color="#1f4e79", linewidth=1.5, label="Histórico")
        ax.plot(x_future, future_with_anchor, color="#c0392b", linewidth=1.5, linestyle="--", label="Previsão Prophet")
        ax.axvline(x=n_hist - 1, color="gray", linestyle=":", linewidth=1)

        ax.set_title(f"Série {row['series']} (RMSE={rmse_val:.3f})", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(n_series, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    output_dir = project_root / "resultados" / "p5_prophet"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "prophet_forecasts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Gráfico salvo em prophet_forecasts.png")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_excel(project_root / "data" / "DadosCompeticao.xlsx")

    print("Treinando modelos Prophet...")
    results = run_prophet(df, test_size=36)

    print("\nResultados por série:")
    print(results[["series", "rmse"]].to_string(index=False))
    print(f"\nWRMSE: {weighted_rmse(results):.4f}")

    print("\nGerando gráficos...")
    plot_forecasts(df, results)
