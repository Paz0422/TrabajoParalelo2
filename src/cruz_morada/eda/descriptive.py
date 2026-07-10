"""Estadística descriptiva y pruebas de asociación."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula medidas de tendencia central, dispersión, asimetría y curtosis."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame()

    desc = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skewness"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    return desc.round(4)


def correlation_with_pvalues(
    df: pd.DataFrame, columns: list[str], method: str = "pearson"
) -> pd.DataFrame:
    """Matriz de correlación (Pearson o Spearman) con p-values."""
    sub = df[columns].dropna()
    n = len(columns)
    test_func = stats.pearsonr if method == "pearson" else stats.spearmanr
    corr = sub.corr(method=method)
    pvalues = pd.DataFrame(np.ones((n, n)), index=columns, columns=columns)

    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            if i >= j:
                continue
            r, p = test_func(sub[col_i], sub[col_j])
            corr.loc[col_i, col_j] = r
            corr.loc[col_j, col_i] = r
            pvalues.loc[col_i, col_j] = p
            pvalues.loc[col_j, col_i] = p

    return corr.round(4), pvalues.round(4)


def chi_square_canal_local(df: pd.DataFrame) -> dict:
    """Chi-cuadrado de independencia entre CANAL y LOCAL."""
    contingency = pd.crosstab(df["CANAL"], df["LOCAL"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "significant_005": p < 0.05,
    }


def anova_monto_by_canal(df: pd.DataFrame) -> dict:
    """ANOVA: diferencias en MONTO APLICADO entre canales."""
    groups = [g["MONTO APLICADO"].dropna().values for _, g in df.groupby("CANAL")]
    if len(groups) < 2:
        return {"error": "Se requieren al menos 2 canales"}

    f_stat, p = stats.f_oneway(*groups)
    return {"f_statistic": float(f_stat), "p_value": float(p), "significant_005": p < 0.05}


def normality_tests(series: pd.Series, max_sample: int = 5000) -> dict:
    """Shapiro-Wilk y Kolmogorov-Smirnov sobre muestra."""
    sample = series.dropna()
    if len(sample) > max_sample:
        sample = sample.sample(max_sample, random_state=42)

    shapiro_stat, shapiro_p = stats.shapiro(sample)
    ks_stat, ks_p = stats.kstest(sample, stats.norm(loc=sample.mean(), scale=sample.std()).cdf)

    return {
        "shapiro": {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)},
        "kolmogorov_smirnov": {"statistic": float(ks_stat), "p_value": float(ks_p)},
    }
