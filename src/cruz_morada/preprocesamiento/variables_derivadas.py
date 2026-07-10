"""Variables derivadas y transformaciones."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


MIN_EDAD_PLAUSIBLE = 0
MAX_EDAD_PLAUSIBLE = 100


def compute_edad_mediana_global(df: pd.DataFrame) -> float:
    """
    Calcula, de forma secuencial y vectorizada, la mediana global de EDAD
    plausible sobre el DataFrame completo (ya limpio).

    Se calcula una sola vez y se pasa como constante a create_derived_features
    cuando esta corre en paralelo por partición (ver map_partitions_by_key en
    ParallelProcessor): si cada partición calculara su propia mediana local, el
    valor imputado dependería de qué filas cayeron en cada partición, dejando
    de ser reproducible/consistente con la corrida secuencial. El cálculo en sí
    es barato incluso con millones de filas (una resta de fechas vectorizada),
    por lo que no vale la pena paralelizarlo.
    """
    edad = ((df["FECHA"] - df["FECHA NACIMIENTO"]).dt.days / 365.25).round(1)
    implausible = (edad < MIN_EDAD_PLAUSIBLE) | (edad > MAX_EDAD_PLAUSIBLE)
    plausible = edad[~implausible]
    return float(plausible.median()) if len(plausible) else float(edad.median())


def create_derived_features(
    df: pd.DataFrame,
    edad_mediana_global: float | None = None,
) -> pd.DataFrame:
    """
    Crea variables derivadas requeridas por el enunciado.

    Args:
        edad_mediana_global: si se ejecuta en paralelo por partición, la mediana
            de EDAD debe calcularse una sola vez sobre el DataFrame completo
            (con compute_edad_mediana_global) y pasarse aquí; si se omite, se
            calcula localmente sobre *df* (correcto solo en modo secuencial).
    """
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
            median_edad = (
                edad_mediana_global if edad_mediana_global is not None else edad[~implausible].median()
            )
            edad = edad.mask(implausible, median_edad)
            logger.info(
                "EDAD: %d registros fuera de rango plausible [%d, %d] imputados con mediana %.1f",
                n_implausible, MIN_EDAD_PLAUSIBLE, MAX_EDAD_PLAUSIBLE, median_edad,
            )
        df["EDAD"] = edad

    # FRECUENCIA COMPRA e ITEMS POR BOLETA agrupan por CODIGO CLIENTE y BOLETA
    # respectivamente. Si esta función corre en paralelo por partición, hay que
    # particionar por hash(CODIGO CLIENTE) (ParallelProcessor.map_partitions_by_key)
    # y no por columnas no relacionadas (LOCAL, fecha): de lo contrario un mismo
    # cliente podría quedar repartido entre procesos y su frecuencia de compra
    # saldría subestimada (cada partición solo vería una parte de sus boletas).
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
