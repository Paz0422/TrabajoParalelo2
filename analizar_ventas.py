#!/usr/bin/env python3
"""
Análisis Estadístico de Datos de Ventas - Cruz Morada
Computación Paralela y Distribuida - UTEM

Uso:
    python analizar_ventas.py --csv data/ventas_completas.csv
    python analizar_ventas.py --csv data/ventas_completas.csv --dask --workers 4
    set CPYD_SEED=42 && python analizar_ventas.py --csv data/ventas_completas.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Agregar src al path para imports locales
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from cruz_morada.configuracion import OUTPUT_DIR, SEED
from cruz_morada.carga_datos import load_csv
from cruz_morada.paralelo import ParallelProcessor
from cruz_morada.preprocesamiento import (
    clean_data,
    compute_edad_mediana_global,
    create_derived_features,
    report_missing_values,
    summarize_outliers,
    test_mcar_little,
)
from cruz_morada.analisis_exploratorio import (
    compute_descriptive_stats,
    correlation_with_pvalues,
    daily_sales_series,
    generate_eda_plots,
    normality_tests,
    plot_acf_pacf,
    seasonal_decomposition,
)
from cruz_morada.analisis_exploratorio.estadistica_descriptiva import anova_monto_by_canal, chi_square_canal_local
from cruz_morada.inferencia import run_hypothesis_tests, run_regression_model, run_clustering_model


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis estadístico paralelo de ventas - Cruz Morada",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Ruta al archivo ventas_completas.csv",
    )
    parser.add_argument(
        "--dask",
        action="store_true",
        help="Usar Dask para carga lazy del CSV",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Limitar filas (útil para pruebas)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de workers para procesamiento paralelo",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Omitir generación de gráficos",
    )
    parser.add_argument(
        "--parallel-preprocess",
        action="store_true",
        help=(
            "Paralelizar limpieza y transformación por partición (además del "
            "cálculo de estadísticos por LOCAL, que siempre es paralelo). "
            "Por defecto se hace secuencial: en las mediciones sobre el "
            "dataset completo, el overhead de serializar particiones grandes "
            "entre procesos resultó más lento que pandas vectorizado en un "
            "solo proceso (ver sección de dificultades del informe)."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "eda", "inference"],
        default="all",
        help="Etapa a ejecutar",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def run_preprocess(df, processor: ParallelProcessor, parallel_preprocess: bool = False) -> tuple:
    """Parte 1: Datos y Preprocesamiento (20%)."""
    logger = logging.getLogger("pipeline.preprocess")
    results = {}

    logger.info("=== PREPROCESAMIENTO ===")
    results["missing_before"] = report_missing_values(df).to_dict()
    results["mcar_test"] = test_mcar_little(df)

    if parallel_preprocess:
        # --- Limpieza en paralelo ---
        # La validez de cada fila (UNIDADES>0, BOLETA>0, etc.) es independiente
        # del resto, así que cualquier partición arbitraria es correcta aquí.
        t0 = time.perf_counter()
        cleaned_parts = processor.map_partitions(df, clean_data)
        df_clean = pd.concat(cleaned_parts, ignore_index=True)
        logger.info(
            "Limpieza en paralelo (%d particiones, %d workers): %.2fs",
            len(cleaned_parts), processor.n_workers, time.perf_counter() - t0,
        )
    else:
        t0 = time.perf_counter()
        df_clean = clean_data(df)
        logger.info("Limpieza secuencial (pandas vectorizado): %.2fs", time.perf_counter() - t0)

    results["missing_after"] = report_missing_values(df_clean).to_dict()

    # FRECUENCIA COMPRA e ITEMS POR BOLETA calculan groupby(CODIGO CLIENTE) y
    # groupby(BOLETA) internamente. Una boleta pertenece siempre a un único
    # cliente, así que particionar por hash(CODIGO CLIENTE) garantiza que
    # ambos agrupamientos queden completos dentro de cada partición: a
    # diferencia de particionar por LOCAL o por fecha (donde un mismo cliente
    # puede comprar en distintos locales/días y quedar repartido entre
    # procesos), aquí ningún cliente ni boleta se reparte entre particiones.
    # La mediana de EDAD se calcula una sola vez sobre el DataFrame completo
    # (ver compute_edad_mediana_global) para que la imputación sea la misma
    # sin importar en qué partición caiga cada fila.
    if parallel_preprocess:
        edad_mediana_global = compute_edad_mediana_global(df_clean)
        t0 = time.perf_counter()
        transformed_parts = processor.map_partitions_by_key(
            df_clean,
            create_derived_features,
            key_col="CODIGO CLIENTE",
            edad_mediana_global=edad_mediana_global,
        )
        df_clean = pd.concat(transformed_parts, ignore_index=True)
        logger.info(
            "Transformación en paralelo (%d particiones, %d workers): %.2fs",
            len(transformed_parts), processor.n_workers, time.perf_counter() - t0,
        )
    else:
        t0 = time.perf_counter()
        df_clean = create_derived_features(df_clean)
        logger.info("Transformación secuencial (pandas vectorizado): %.2fs", time.perf_counter() - t0)

    results["outliers_monto"] = summarize_outliers(df_clean, "MONTO APLICADO")
    results["outliers_unidades"] = summarize_outliers(df_clean, "UNIDADES")

    # Estadísticos paralelos por LOCAL
    parallel_stats = processor.map_partitions(
        df_clean, ParallelProcessor.sales_chunk_stats, partition_col="LOCAL"
    )
    results["parallel_stats_by_local"] = processor.combine_stats(parallel_stats)

    return df_clean, results


def run_eda(df) -> dict:
    """Parte 2: Análisis Exploratorio Estadístico (30%)."""
    logger = logging.getLogger("pipeline.eda")
    results = {}

    logger.info("=== ANÁLISIS EXPLORATORIO ===")
    results["descriptive_stats"] = compute_descriptive_stats(df).to_dict()

    cols = [c for c in ["UNIDADES", "MONTO APLICADO", "PORCENTAJE DESCUENTO"] if c in df.columns]

    normality = {col: normality_tests(df[col]) for col in cols}
    results["normality_tests"] = normality
    all_normal = all(
        normality[col]["shapiro"]["p_value"] > 0.05 for col in cols
    ) if cols else False
    corr_method = "pearson" if all_normal else "spearman"
    results["correlation_method"] = corr_method
    logger.info(
        "Normalidad %s en variables numéricas -> usando correlación de %s",
        "detectada" if all_normal else "no detectada",
        corr_method,
    )

    if len(cols) >= 2:
        corr, pvals = correlation_with_pvalues(df, cols, method=corr_method)
        results["correlation"] = corr.to_dict()
        results["correlation_pvalues"] = pvals.to_dict()

    results["chi2_canal_local"] = chi_square_canal_local(df)
    results["anova_monto_canal"] = anova_monto_by_canal(df)

    if "CANAL" in df.columns:
        conteo_canal = df["CANAL"].value_counts()
        results["distribucion_canal"] = {
            "conteo": conteo_canal.to_dict(),
            "porcentaje": (conteo_canal / len(df) * 100).round(2).to_dict(),
        }

    if "HORA" in df.columns:
        ventas_por_hora = df.groupby("HORA")["MONTO APLICADO"].sum()
        results["hora_mayor_venta"] = int(ventas_por_hora.idxmax())
        results["hora_menor_venta"] = int(ventas_por_hora.idxmin())

    daily = daily_sales_series(df)
    results["daily_sales_summary"] = {
        "mean": float(daily.mean()),
        "std": float(daily.std()),
        "max_date": str(daily.idxmax()),
    }

    return results, daily


def run_inference(df) -> dict:
    """Parte 3: Inferencia Estadística (30%)."""
    logger = logging.getLogger("pipeline.inference")
    results = {}

    logger.info("=== INFERENCIA Y MODELADO ===")
    results["hypothesis_tests"] = run_hypothesis_tests(df)
    results["regression"] = run_regression_model(df)
    results["clustering"] = run_clustering_model(df)

    return results


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("main")

    logger.info("Semilla CPYD_SEED = %d", SEED)
    logger.info("Archivo CSV: %s", args.csv)

    # 1. Carga
    df = load_csv(args.csv, use_dask=args.dask, n_rows=args.n_rows)
    all_results: dict = {"seed": SEED, "n_rows": len(df)}

    # El pool de procesos se crea una sola vez y se reutiliza en todas las
    # etapas (limpieza, transformación, estadísticos por LOCAL) en vez de
    # crear y destruir workers en cada llamada a map_partitions.
    with ParallelProcessor(n_workers=args.workers) as processor:
        # 2. Preprocesamiento
        if args.stage in ("all", "preprocess"):
            df, prep_results = run_preprocess(df, processor, args.parallel_preprocess)
            all_results["preprocessing"] = prep_results

        # 3. EDA
        daily = None
        if args.stage in ("all", "eda", "inference"):
            if args.stage == "inference" and "preprocessing" not in all_results:
                df, _ = run_preprocess(df, processor, args.parallel_preprocess)

            if args.stage in ("all", "eda"):
                eda_results, daily = run_eda(df)
                all_results["eda"] = eda_results

                if not args.skip_plots:
                    generate_eda_plots(df)
                    if daily is not None and len(daily) > 14:
                        seasonal_decomposition(daily)
                        plot_acf_pacf(daily)

        # 4. Inferencia
        if args.stage in ("all", "inference"):
            if "eda" not in all_results and args.stage == "inference":
                df, _ = run_preprocess(df, processor, args.parallel_preprocess)
            all_results["inference"] = run_inference(df)

    # Guardar resultados JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "resultados.json"

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=_json_default)

    logger.info("Resultados guardados en %s", results_path)
    logger.info("Análisis completado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
