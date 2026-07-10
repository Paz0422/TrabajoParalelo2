"""Carga eficiente del archivo CSV de ventas."""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import pandas as pd

from cruz_morada.configuracion import CHUNK_SIZE, COLUMNS

logger = logging.getLogger(__name__)

# datos personales que no se usan en el análisis; se descartan al cargar para ahorrar memoria
PII_DROP_COLUMNS = ["NOMBRES", "APELLIDOS"]


def _detect_separator(csv_path: Path) -> str:
    """Detecta si el CSV usa ',' o ';' inspeccionando la primera línea."""
    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt", encoding="utf-8") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    logger.info("Separador detectado: '%s'", sep)
    return sep


def load_csv(
    csv_path: Path,
    use_dask: bool = False,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Carga el CSV (o .csv.gz) por fragmentos; con use_dask=True usa lectura lazy.

    n_rows sirve para probar rápido con un subset. Devuelve todas las columnas
    del enunciado menos las de PII_DROP_COLUMNS.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {csv_path}\n"
            "Descarga ventas_completas.csv.gz desde Google Drive y colócalo en data/"
        )

    sep = _detect_separator(csv_path)

    if use_dask:
        return _load_with_dask(csv_path, sep, n_rows)

    return _load_with_chunks(csv_path, sep, n_rows)


def _load_with_chunks(csv_path: Path, sep: str, n_rows: int | None) -> pd.DataFrame:
    """Lee el CSV en chunks y concatena."""
    chunks: list[pd.DataFrame] = []
    total_rows = 0

    logger.info("Cargando %s por fragmentos (chunk_size=%d)...", csv_path.name, CHUNK_SIZE)

    string_dtype_cols = ["CANAL", "PRODUCTO", "CODIGO CLIENTE", "RUN CLIENTE", "NOMBRES", "APELLIDOS"]

    for chunk in pd.read_csv(
        csv_path,
        sep=sep,
        chunksize=CHUNK_SIZE,
        dtype={c: "string" for c in string_dtype_cols},
        low_memory=False,
    ):
        chunk = chunk.drop(columns=[c for c in PII_DROP_COLUMNS if c in chunk.columns])
        chunks.append(chunk)
        total_rows += len(chunk)
        if n_rows and total_rows >= n_rows:
            break

    df = pd.concat(chunks, ignore_index=True)
    if n_rows:
        df = df.head(n_rows)

    _validate_columns(df)
    logger.info("Carga completada: %d filas, %d columnas", len(df), len(df.columns))
    return df


def _load_with_dask(csv_path: Path, sep: str, n_rows: int | None) -> pd.DataFrame:
    """Lee con Dask y materializa a pandas."""
    import dask.dataframe as dd

    logger.info("Cargando %s con Dask...", csv_path.name)
    ddf = dd.read_csv(csv_path, sep=sep, assume_missing=True, blocksize=None if csv_path.suffix == ".gz" else "64MB")
    ddf = ddf.drop(columns=[c for c in PII_DROP_COLUMNS if c in ddf.columns])
    if n_rows:
        df = ddf.head(n_rows, npartitions=-1)
    else:
        df = ddf.compute()
    _validate_columns(df)
    logger.info("Carga Dask completada: %d filas", len(df))
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    expected = [c for c in COLUMNS if c not in PII_DROP_COLUMNS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning("Columnas faltantes respecto al enunciado: %s", missing)
