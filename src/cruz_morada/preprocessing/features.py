"""Variables derivadas y transformaciones."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


MIN_EDAD_PLAUSIBLE = 0
MAX_EDAD_PLAUSIBLE = 100


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas requeridas por el enunciado."""
    df = df.copy()

    df["MONTO POR UNIDAD"] = df["MONTO APLICADO"] / df["UNIDADES"]

    if "FECHA NACIMIENTO" in df.columns and "FECHA" in df.columns:
        edad = ((df["FECHA"] - df["FECHA NACIMIENTO"]).dt.days / 365.25).round(1)

        # FECHA NACIMIENTO trae registros corruptos (fechas futuras a la compra o
        # muy antiguas) que generan edades imposibles (ej. negativas o > 100 años).
        # Se tratan como error de registro: se invalidan y se imputan con la mediana
        # de las edades plausibles, en vez de dejar que distorsionen el análisis.
        implausible = (edad < MIN_EDAD_PLAUSIBLE) | (edad > MAX_EDAD_PLAUSIBLE)
        n_implausible = int(implausible.sum())
        if n_implausible:
            median_edad = edad[~implausible].median()
            edad = edad.mask(implausible, median_edad)
            logger.info(
                "EDAD: %d registros fuera de rango plausible [%d, %d] imputados con mediana %.1f",
                n_implausible, MIN_EDAD_PLAUSIBLE, MAX_EDAD_PLAUSIBLE, median_edad,
            )
        df["EDAD"] = edad

    if "CODIGO CLIENTE" in df.columns:
        freq = df.groupby("CODIGO CLIENTE")["BOLETA"].transform("nunique")
        df["FRECUENCIA COMPRA"] = freq

    # UNIDADES es constante (=1) en la totalidad de los registros: el dataset está
    # a nivel de línea de producto, no de cantidad por línea. Se deriva un proxy real
    # de volumen de compra: cantidad de líneas (productos distintos) por boleta.
    if "BOLETA" in df.columns:
        df["ITEMS POR BOLETA"] = df.groupby("BOLETA")["BOLETA"].transform("size")

    df["HORA"] = df["FECHA"].dt.hour
    df["DIA SEMANA"] = df["FECHA"].dt.dayofweek
    df["MES"] = df["FECHA"].dt.month
    df["ANIO"] = df["FECHA"].dt.year
    df["ES_FIN_DE_SEMANA"] = df["DIA SEMANA"].isin([5, 6])

    logger.info(
        "Variables derivadas creadas: MONTO POR UNIDAD, EDAD, FRECUENCIA COMPRA, "
        "ITEMS POR BOLETA, temporales"
    )
    return df


def standardize_numeric(
    df: pd.DataFrame,
    columns: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Estandariza columnas numéricas documentando parámetros del scaler."""
    df = df.copy()
    scaler = scaler or StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler
