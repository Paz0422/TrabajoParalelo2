"""Análisis de series temporales."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from cruz_morada.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def daily_sales_series(df: pd.DataFrame) -> pd.Series:
    """Agrega ventas diarias (suma de MONTO APLICADO)."""
    daily = (
        df.set_index("FECHA")
        .resample("D")["MONTO APLICADO"]
        .sum()
        .fillna(0)
    )
    return daily


def seasonal_decomposition(
    series: pd.Series,
    period: int = 7,
    output_dir: Path | None = None,
) -> object:
    """Descomposición aditiva: tendencia, estacionalidad y residuales."""
    if len(series) < 2 * period:
        logger.warning("Serie muy corta para descomposición estacional")
        return None

    result = seasonal_decompose(series, model="additive", period=period)
    out = output_dir or OUTPUT_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    fig = result.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(out / "seasonal_decomposition.png", dpi=150)
    plt.close()

    return result


def plot_acf_pacf(series: pd.Series, output_dir: Path | None = None) -> None:
    """Gráficos ACF y PACF para dependencia temporal."""
    out = output_dir or OUTPUT_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series.dropna(), ax=axes[0], lags=40)
    plot_pacf(series.dropna(), ax=axes[1], lags=40)
    axes[0].set_title("Autocorrelación (ACF)")
    axes[1].set_title("Autocorrelación parcial (PACF)")
    plt.tight_layout()
    plt.savefig(out / "acf_pacf.png", dpi=150)
    plt.close()
