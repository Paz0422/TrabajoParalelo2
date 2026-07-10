"""Pruebas de hipótesis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


def run_hypothesis_tests(df: pd.DataFrame) -> dict:
    """
    Ejecuta pruebas de hipótesis del enunciado + 3 hipótesis propias.

    H1: Ticket promedio APP > WEB (t-test o Mann-Whitney)
    H2: % descuento afecta el volumen de compra (regresión lineal). UNIDADES es
        constante (=1) en el 100% de los registros reales (dataset a nivel de línea
        de producto), por lo que se usa ITEMS POR BOLETA como proxy real de volumen.
    H3-H5: Hipótesis propias.
    """
    results: dict = {}

    # H1: ticket promedio APP vs WEB (ejemplo del enunciado)
    results["H1_ticket_app_vs_web"] = _test_ticket_by_channel(df, "APP", "WEB")

    # H2: descuento vs volumen de compra (ejemplo adaptado)
    results["H2_descuento_vs_volumen"] = _test_discount_affects_units(df)

    # H3-H5: propias
    results["H3_monto_finde_vs_semana"] = _test_weekend_vs_weekday(df)
    results["H4_edad_vs_monto"] = _test_age_vs_amount(df)
    results["H5_genero_vs_monto"] = _test_gender_vs_amount(df)

    return results


def _test_ticket_by_channel(
    df: pd.DataFrame,
    channel_a: str,
    channel_b: str,
) -> dict:
    """Compara ticket promedio entre dos canales."""
    mask_a = df["CANAL"].str.upper() == channel_a.upper()
    mask_b = df["CANAL"].str.upper() == channel_b.upper()
    a = df.loc[mask_a, "MONTO APLICADO"].dropna()
    b = df.loc[mask_b, "MONTO APLICADO"].dropna()

    if len(a) < 2 or len(b) < 2:
        return {"error": f"Datos insuficientes para {channel_a} vs {channel_b}"}

    # Verificar normalidad para elegir test
    _, p_norm_a = stats.shapiro(a.sample(min(5000, len(a)), random_state=42))
    _, p_norm_b = stats.shapiro(b.sample(min(5000, len(b)), random_state=42))
    normal = p_norm_a > 0.05 and p_norm_b > 0.05

    if normal:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        test_name = "Welch t-test"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Mann-Whitney U"

    return {
        "hypothesis": f"Ticket promedio {channel_a} vs {channel_b}",
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "significant_005": p < 0.05,
    }


def _test_discount_affects_units(df: pd.DataFrame) -> dict:
    """Regresión lineal: ITEMS POR BOLETA ~ PORCENTAJE DESCUENTO.

    UNIDADES es constante (=1) en el 100% de los registros reales, por lo que no
    aporta varianza para esta prueba; se usa ITEMS POR BOLETA (líneas de producto
    por transacción) como proxy real del volumen de compra.
    """
    volume_col = "ITEMS POR BOLETA" if "ITEMS POR BOLETA" in df.columns else "UNIDADES"
    sub = df[[volume_col, "PORCENTAJE DESCUENTO"]].dropna()
    if len(sub) < 10 or sub[volume_col].nunique() < 2:
        return {"error": "Datos insuficientes o sin varianza"}

    slope, intercept, r, p, se = stats.linregress(
        sub["PORCENTAJE DESCUENTO"], sub[volume_col]
    )
    return {
        "hypothesis": "El % de descuento afecta el volumen de compra (items por boleta)",
        "test": "Regresión lineal simple",
        "variable_dependiente": volume_col,
        "slope": float(slope),
        "r_squared": float(r ** 2),
        "p_value": float(p),
        "significant_005": p < 0.05,
    }


def _test_age_vs_amount(df: pd.DataFrame) -> dict:
    """H3 propia: correlación entre EDAD y MONTO APLICADO."""
    if "EDAD" not in df.columns:
        return {"error": "Variable EDAD no disponible"}
    sub = df[["EDAD", "MONTO APLICADO"]].dropna()
    r, p = stats.pearsonr(sub["EDAD"], sub["MONTO APLICADO"])
    return {
        "hypothesis": "La edad del cliente se correlaciona con el monto de compra",
        "test": "Pearson",
        "correlation": float(r),
        "p_value": float(p),
        "significant_005": p < 0.05,
    }


def _test_weekend_vs_weekday(df: pd.DataFrame) -> dict:
    """H3 propia: ¿El ticket promedio difiere entre fin de semana y días de semana?"""
    if "ES_FIN_DE_SEMANA" not in df.columns:
        return {"error": "Variable ES_FIN_DE_SEMANA no disponible"}

    a = df.loc[df["ES_FIN_DE_SEMANA"], "MONTO APLICADO"].dropna()
    b = df.loc[~df["ES_FIN_DE_SEMANA"], "MONTO APLICADO"].dropna()
    if len(a) < 2 or len(b) < 2:
        return {"error": "Datos insuficientes"}

    _, p_norm_a = stats.shapiro(a.sample(min(5000, len(a)), random_state=42))
    _, p_norm_b = stats.shapiro(b.sample(min(5000, len(b)), random_state=42))
    normal = p_norm_a > 0.05 and p_norm_b > 0.05

    if normal:
        stat, p = stats.ttest_ind(a, b, equal_var=False)
        test_name = "Welch t-test"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Mann-Whitney U"

    return {
        "hypothesis": "El ticket promedio (MONTO APLICADO) difiere entre fin de semana y día de semana",
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "mean_finde": float(a.mean()),
        "mean_semana": float(b.mean()),
        "significant_005": p < 0.05,
    }


def _test_gender_vs_amount(df: pd.DataFrame) -> dict:
    """H5 propia: diferencia de MONTO APLICADO por GÉNERO (ANOVA)."""
    if "GENERO" not in df.columns:
        return {"error": "Variable GENERO no disponible"}
    groups = [g["MONTO APLICADO"].dropna().values for _, g in df.groupby("GENERO")]
    if len(groups) < 2:
        return {"error": "Solo un grupo de género"}
    f_stat, p = stats.f_oneway(*groups)
    return {
        "hypothesis": "El género del cliente influye en el monto promedio de compra",
        "test": "ANOVA one-way",
        "f_statistic": float(f_stat),
        "p_value": float(p),
        "significant_005": p < 0.05,
    }
