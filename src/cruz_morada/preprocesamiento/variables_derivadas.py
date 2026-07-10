"""Variables derivadas y transformaciones."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


MIN_EDAD_PLAUSIBLE = 0
MAX_EDAD_PLAUSIBLE = 100


def compute_edad_mediana_global(df: pd.DataFrame) -> float:
    """Mediana de EDAD plausible sobre todo el df (una sola vez, no por partición).

    Se le pasa a create_derived_features cuando corre en paralelo: si cada
    partición sacara su propia mediana, la imputación dependería de qué filas
    le tocaron a cada una. El cálculo es rápido igual (resta de fechas
    vectorizada), así que tampoco vale la pena paralelizarlo.
    """
    edad = ((df["FECHA"] - df["FECHA NACIMIENTO"]).dt.days / 365.25).round(1)
    implausible = (edad < MIN_EDAD_PLAUSIBLE) | (edad > MAX_EDAD_PLAUSIBLE)
    plausible = edad[~implausible]
    return float(plausible.median()) if len(plausible) else float(edad.median())


def create_derived_features(
    df: pd.DataFrame,
    edad_mediana_global: float | None = None,
) -> pd.DataFrame:
    """Crea las variables derivadas del enunciado.

    edad_mediana_global se pasa cuando esto corre en paralelo por partición
    (ver compute_edad_mediana_global); si se omite, calcula la mediana sobre
    el propio *df*, que solo es correcto en modo secuencial.
    """
    df = df.copy()

    df["MONTO POR UNIDAD"] = df["MONTO APLICADO"] / df["UNIDADES"]

    if "FECHA NACIMIENTO" in df.columns and "FECHA" in df.columns:
        edad = ((df["FECHA"] - df["FECHA NACIMIENTO"]).dt.days / 365.25).round(1)

        # hay FECHA NACIMIENTO corrupta que da edades negativas o >100 años;
        # se trata como error de registro y se imputa con la mediana
        implausible = (edad < MIN_EDAD_PLAUSIBLE) | (edad > MAX_EDAD_PLAUSIBLE)
        n_implausible = int(implausible.sum())
        if n_implausible:
            median_edad = (
                edad_mediana_global if edad_mediana_global is not None else edad[~implausible].median()
            )
            edad = edad.mask(implausible, median_edad)
            logger.info(
                "EDAD: %d registros fuera de rango plausible [%d, %d] imputados con mediana %.1f",
                n_implausible, MIN_EDAD_PLAUSIBLE, MAX_EDAD_PLAUSIBLE, median_edad,
            )
        df["EDAD"] = edad

    # Ojo: esto agrupa por CODIGO CLIENTE, así que en paralelo hay que
    # particionar por esa clave (ver map_partitions_by_key), no por LOCAL/fecha.
    if "CODIGO CLIENTE" in df.columns:
        freq = df.groupby("CODIGO CLIENTE")["BOLETA"].transform("nunique")
        df["FRECUENCIA COMPRA"] = freq

    # UNIDADES es siempre 1 (dataset a nivel de línea de producto), así que
    # esto es el proxy real de volumen: líneas distintas por boleta
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
