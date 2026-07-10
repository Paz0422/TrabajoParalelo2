"""Detección de outliers."""

from __future__ import annotations

import pandas as pd


def detect_outliers_iqr(
    series: pd.Series,
    factor: float = 1.5,
) -> pd.Series:
    """Marca outliers usando el método IQR (robusto)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """Marca outliers usando Z-score."""
    z = (series - series.mean()) / series.std(ddof=0)
    return z.abs() > threshold


def summarize_outliers(df: pd.DataFrame, column: str) -> dict:
    """Resume outliers detectados por IQR y Z-score."""
    if column not in df.columns:
        return {"error": f"Columna {column} no encontrada"}

    s = df[column].dropna()
    iqr_mask = detect_outliers_iqr(s)
    z_mask = detect_outliers_zscore(s)

    return {
        "column": column,
        "n_iqr_outliers": int(iqr_mask.sum()),
        "pct_iqr_outliers": round(iqr_mask.mean() * 100, 2),
        "n_zscore_outliers": int(z_mask.sum()),
        "pct_zscore_outliers": round(z_mask.mean() * 100, 2),
    }
