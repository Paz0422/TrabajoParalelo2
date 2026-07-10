"""Limpieza y tratamiento de valores faltantes."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from cruz_morada.configuracion import SEED

logger = logging.getLogger(__name__)
RNG = np.random.default_rng(SEED)


def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Genera reporte de valores ausentes por columna."""
    report = pd.DataFrame(
        {
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "dtype": df.dtypes.astype(str),
        }
    )
    report = report[report["n_missing"] > 0].sort_values("pct_missing", ascending=False)
    return report


def test_mcar_little(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """
    Prueba aproximada de MCAR usando correlación entre indicadores de missingness.
    Valores altos sugieren que los faltantes no son completamente aleatorios.
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    if not cols:
        return {"message": "No hay columnas numéricas para evaluar MCAR"}

    indicators = df[cols].isna().astype(int)
    if indicators.sum().sum() == 0:
        return {"message": "No hay valores faltantes"}

    corr = indicators.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    max_corr = upper.stack().max() if not upper.stack().empty else 0.0

    return {
        "max_missing_indicator_correlation": float(max_corr),
        "interpretation": (
            "Correlación baja sugiere MCAR plausible"
            if max_corr < 0.3
            else "Posible patrón sistemático en faltantes (MNAR/MAR)"
        ),
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de limpieza: tipos, faltantes e imputación justificada.

    Estrategia base (ajustar según informe):
    - PORCENTAJE DESCUENTO: imputar 0 (sin descuento)
    - FECHA NACIMIENTO: imputar mediana de edad
    - Eliminar filas con MONTO APLICADO o UNIDADES inválidos
    """
    df = df.copy()

    # Normalizar nombre de columna GÉNERO -> GENERO
    if "GÉNERO" in df.columns and "GENERO" not in df.columns:
        df = df.rename(columns={"GÉNERO": "GENERO"})

    # Conversión de tipos
    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce", utc=True).dt.tz_localize(None)
    df["FECHA NACIMIENTO"] = pd.to_datetime(df["FECHA NACIMIENTO"], errors="coerce")

    for col in ["UNIDADES", "MONTO APLICADO", "PORCENTAJE DESCUENTO", "LOCAL", "SKU"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar registros críticos inválidos (incluye BOLETA=0, error de registro:
    # un número de boleta 0 no corresponde a un documento tributario real)
    before = len(df)
    df = df.dropna(subset=["MONTO APLICADO", "UNIDADES", "FECHA"])
    df = df[(df["UNIDADES"] > 0) & (df["MONTO APLICADO"] >= 0)]
    if "BOLETA" in df.columns:
        df = df[df["BOLETA"] > 0]
    logger.info("Filas eliminadas por valores inválidos: %d", before - len(df))

    # Imputación de faltantes
    if "PORCENTAJE DESCUENTO" in df.columns:
        df["PORCENTAJE DESCUENTO"] = df["PORCENTAJE DESCUENTO"].fillna(0.0)

        # PORCENTAJE DESCUENTO está definido en el enunciado como 0 a 1: valores
        # fuera de ese rango (ej. 1.17) son error de registro, no un descuento real.
        before_pct = len(df)
        df = df[(df["PORCENTAJE DESCUENTO"] >= 0) & (df["PORCENTAJE DESCUENTO"] <= 1)]
        if before_pct - len(df):
            logger.info(
                "Filas eliminadas por PORCENTAJE DESCUENTO fuera de rango [0,1]: %d",
                before_pct - len(df),
            )

    if "FECHA NACIMIENTO" in df.columns and df["FECHA NACIMIENTO"].isna().any():
        median_birth = df["FECHA NACIMIENTO"].median()
        df["FECHA NACIMIENTO"] = df["FECHA NACIMIENTO"].fillna(median_birth)
        logger.info("FECHA NACIMIENTO imputada con mediana: %s", median_birth)

    # Imputación estocástica opcional para GENERO (con semilla fija)
    if "GENERO" in df.columns and df["GENERO"].isna().any():
        mode_gen = df["GENERO"].mode()
        fill_value = mode_gen.iloc[0] if len(mode_gen) else 1
        mask = df["GENERO"].isna()
        df.loc[mask, "GENERO"] = fill_value

    return df.reset_index(drop=True)
