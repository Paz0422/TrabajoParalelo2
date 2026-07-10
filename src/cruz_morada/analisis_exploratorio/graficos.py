"""Visualizaciones obligatorias del EDA."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from cruz_morada.configuracion import OUTPUT_DIR, SEED

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")


def generate_eda_plots(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Genera histogramas, boxplots y matriz de correlación."""
    out = output_dir or OUTPUT_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    numeric_cols = ["UNIDADES", "MONTO APLICADO", "PORCENTAJE DESCUENTO", "MONTO POR UNIDAD"]
    available = [c for c in numeric_cols if c in df.columns]

    _plot_histograms_with_density(df, available, out)
    _plot_boxplots_by_channel(df, out)
    _plot_correlation_heatmap(df, available, out)
    _plot_ventas_por_hora(df, out)
    _plot_boxplot_edad_por_genero(df, out)

    logger.info("Gráficos EDA guardados en %s", out)
    return out


def _plot_histograms_with_density(df: pd.DataFrame, columns: list[str], out: Path) -> None:
    for col in columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, stat="density")
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        plt.tight_layout()
        plt.savefig(out / f"hist_{col.replace(' ', '_').lower()}.png", dpi=150)
        plt.close()


def _plot_boxplots_by_channel(df: pd.DataFrame, out: Path) -> None:
    if "CANAL" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="CANAL", y="MONTO APLICADO", ax=ax)
    ax.set_title("MONTO APLICADO por CANAL")
    plt.tight_layout()
    plt.savefig(out / "boxplot_monto_por_canal.png", dpi=150)
    plt.close()


def _plot_ventas_por_hora(df: pd.DataFrame, out: Path) -> None:
    if "HORA" not in df.columns:
        return
    ventas_hora = df.groupby("HORA")["MONTO APLICADO"].sum().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ventas_hora.index, ventas_hora.values)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("MONTO APLICADO total (CLP)")
    ax.set_title("Ventas totales por hora del día")
    ax.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(out / "ventas_por_hora.png", dpi=150)
    plt.close()


def _plot_boxplot_edad_por_genero(df: pd.DataFrame, out: Path) -> None:
    if "GENERO" not in df.columns or "EDAD" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x="GENERO", y="EDAD", ax=ax)
    ax.set_title("EDAD por GÉNERO")
    plt.tight_layout()
    plt.savefig(out / "boxplot_edad_por_genero.png", dpi=150)
    plt.close()


def _plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], out: Path) -> None:
    if len(columns) < 2:
        return
    sub = df[columns].dropna()
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Matriz de correlación")
    plt.tight_layout()
    plt.savefig(out / "correlation_matrix.png", dpi=150)
    plt.close()
