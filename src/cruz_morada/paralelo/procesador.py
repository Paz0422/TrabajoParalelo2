"""Utilidades de procesamiento paralelo sobre particiones de datos."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, TypeVar

import pandas as pd

from cruz_morada.configuracion import N_WORKERS

T = TypeVar("T")


class ParallelProcessor:
    """Corre funciones sobre particiones de un DataFrame con multiprocessing.

    Usar como context manager (`with ParallelProcessor() as p:`) reutiliza el
    mismo pool de procesos entre llamadas en vez de crear uno nuevo cada vez.
    Sin el `with`, funciona igual pero paga el costo de arrancar/cerrar
    procesos en cada llamada.
    """

    def __init__(self, n_workers: int | None = None) -> None:
        self.n_workers = n_workers or N_WORKERS
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "ParallelProcessor":
        if self.n_workers > 1:
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map_partitions(
        self,
        df: pd.DataFrame,
        func: Callable[..., T],
        partition_col: str | None = None,
        n_partitions: int | None = None,
        **kwargs,
    ) -> list[T]:
        """Aplica *func* en paralelo sobre particiones arbitrarias (o por columna) del DataFrame."""
        partitions = self._split(df, partition_col, n_partitions)
        return self._run(func, partitions, **kwargs)

    def map_partitions_by_key(
        self,
        df: pd.DataFrame,
        func: Callable[..., T],
        key_col: str,
        n_partitions: int | None = None,
        **kwargs,
    ) -> list[T]:
        """Como map_partitions, pero particiona por hash de *key_col*.

        Usar esto cuando *func* haga groupby(key_col) por dentro (ej. contar
        boletas por cliente): si particionamos por otra cosa (LOCAL, fecha),
        las filas de un mismo cliente podrían quedar repartidas entre procesos
        y la agregación saldría subestimada. Se usa pd.util.hash_pandas_object
        en vez del hash() de Python porque este último no es determinista
        entre corridas (depende de PYTHONHASHSEED), y necesitamos que el
        particionado sea reproducible con la semilla CPYD_SEED.
        """
        partitions = self._split_by_hash(df, key_col, n_partitions or self.n_workers)
        return self._run(func, partitions, **kwargs)

    def _run(self, func: Callable[..., T], partitions: list[pd.DataFrame], **kwargs) -> list[T]:
        if len(partitions) <= 1 or self.n_workers == 1:
            return [func(part, **kwargs) for part in partitions]

        if self._executor is not None:
            return self._submit_all(self._executor, func, partitions, **kwargs)

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            return self._submit_all(executor, func, partitions, **kwargs)

    @staticmethod
    def _submit_all(
        executor: ProcessPoolExecutor,
        func: Callable[..., T],
        partitions: list[pd.DataFrame],
        **kwargs,
    ) -> list[T]:
        futures = {executor.submit(func, part, **kwargs): i for i, part in enumerate(partitions)}
        ordered = [None] * len(futures)
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
        return [r for r in ordered if r is not None]

    @staticmethod
    def _split(
        df: pd.DataFrame,
        partition_col: str | None,
        n_partitions: int | None,
    ) -> list[pd.DataFrame]:
        if partition_col and partition_col in df.columns:
            return [group for _, group in df.groupby(partition_col, sort=False)]

        n = n_partitions or N_WORKERS
        chunk_size = max(1, len(df) // n)
        return [
            df.iloc[i : i + chunk_size].copy()
            for i in range(0, len(df), chunk_size)
        ]

    @staticmethod
    def _split_by_hash(df: pd.DataFrame, key_col: str, n_buckets: int) -> list[pd.DataFrame]:
        buckets = pd.util.hash_pandas_object(df[key_col], index=False).to_numpy() % max(1, n_buckets)
        return [group for _, group in df.groupby(buckets, sort=False)]

    @staticmethod
    def sales_chunk_stats(chunk: pd.DataFrame) -> dict:
        """Estadísticos agregados por partición de ventas."""
        return {
            "count": len(chunk),
            "sum_monto": float(chunk["MONTO APLICADO"].sum()),
        }

    @staticmethod
    def combine_stats(stats_list: list[dict]) -> dict:
        """Combina estadísticos agregados de múltiples particiones (conteos, sumas)."""
        if not stats_list:
            return {}
        combined = stats_list[0].copy()
        for stats in stats_list[1:]:
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    combined[key] = combined.get(key, 0) + value
                elif isinstance(value, dict):
                    sub = combined.setdefault(key, {})
                    for k, v in value.items():
                        sub[k] = sub.get(k, 0) + v
        return combined
