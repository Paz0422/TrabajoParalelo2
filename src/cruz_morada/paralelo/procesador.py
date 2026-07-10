"""Utilidades de procesamiento paralelo sobre particiones de datos."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, TypeVar

import pandas as pd

from cruz_morada.configuracion import N_WORKERS

T = TypeVar("T")


class ParallelProcessor:
    """Ejecuta funciones sobre chunks de un DataFrame usando multiprocessing."""

    def __init__(self, n_workers: int | None = None) -> None:
        self.n_workers = n_workers or N_WORKERS

    def map_partitions(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], T],
        partition_col: str | None = None,
        n_partitions: int | None = None,
    ) -> list[T]:
        """Aplica *func* en paralelo sobre particiones del DataFrame."""
        partitions = self._split(df, partition_col, n_partitions)
        if len(partitions) <= 1 or self.n_workers == 1:
            return [func(part) for part in partitions]

        results: list[T] = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(func, part): i for i, part in enumerate(partitions)}
            ordered = [None] * len(futures)
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
            results = [r for r in ordered if r is not None]
        return results

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
