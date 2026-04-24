from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    project_root = resolve_project_root()
    data_path = project_root / "data" / "DadosCompeticao.xlsx"
    preds_path = project_root / "resultados" / "previsoes_12_meses" / "Previsoes_12_meses.xlsx"
    out_dir = project_root / "resultados" / "previsoes_12_meses"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path)
    df_preds = pd.read_excel(preds_path)

    cols = list(df_preds.columns)
    n_series = len(cols)

    # build a monthly timeline for plotting: start at 2000-01-01 for consistency
    # historical length may vary; use 2000-01-01 as anchor
    n_hist = len(df)
    start_date = pd.Timestamp("2000-01-01")
    dates_hist = pd.date_range(start=start_date, periods=n_hist, freq="MS")

    horizon = len(df_preds)
    # determine future dates for horizon relative to last historical date
    future_start = dates_hist[-1] + pd.DateOffset(months=1)
    dates_future = pd.date_range(start=future_start, periods=horizon, freq="MS")

    # Create grid of plots (3 rows x 4 cols -> 12 slots, one will be unused)
    ncols = 4
    nrows = int(np.ceil(n_series / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, col in enumerate(cols):
        ax = axes_flat[idx]
        # historical series
        hist = pd.to_numeric(df[col], errors="coerce").values
        preds = pd.to_numeric(df_preds[col], errors="coerce").values

        ax.plot(dates_hist, hist, label="Histórico", color="#1f77b4")
        ax.plot(dates_future, preds, label="Previsão (12 meses)", color="#d62728", linestyle="--")

        ax.set_title(f"Série {col}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # hide any unused subplots
    for j in range(n_series, len(axes_flat)):
        axes_flat[j].set_visible(False)

    out_file = out_dir / "plots_previsoes_11_series.png"
    fig.suptitle("Histórico + Previsões (12 meses) — 11 Séries", fontsize=14)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    # Save individual plots too
    for idx, col in enumerate(cols):
        fig, ax = plt.subplots(figsize=(8, 4))
        hist = pd.to_numeric(df[col], errors="coerce").values
        preds = pd.to_numeric(df_preds[col], errors="coerce").values
        ax.plot(dates_hist, hist, label="Histórico", color="#1f77b4")
        ax.plot(dates_future, preds, label="Previsão (12 meses)", color="#d62728", linestyle="--")
        ax.set_title(f"Série {col}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(out_dir / f"plot_{col}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Plots salvos em: {out_dir}")


if __name__ == "__main__":
    main()
