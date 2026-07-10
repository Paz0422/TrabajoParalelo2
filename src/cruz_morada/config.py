"""Configuración global del proyecto."""

from __future__ import annotations

import os
from pathlib import Path

# Semilla de reproducibilidad (variable de entorno CPYD_SEED)
SEED: int = int(os.getenv("CPYD_SEED", "42"))

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CSV = DATA_DIR / "ventas_completas.csv.gz"

# Columnas del dataset según especificación del enunciado
COLUMNS = [
    "FECHA",
    "CANAL",
    "SKU",
    "PRODUCTO",
    "UNIDADES",
    "PORCENTAJE DESCUENTO",
    "MONTO APLICADO",
    "BOLETA",
    "LOCAL",
    "CODIGO CLIENTE",
    "RUN CLIENTE",
    "NOMBRES",
    "APELLIDOS",
    "FECHA NACIMIENTO",
    "GENERO",
]

NUMERIC_COLUMNS = [
    "SKU",
    "UNIDADES",
    "PORCENTAJE DESCUENTO",
    "MONTO APLICADO",
    "BOLETA",
    "LOCAL",
    "GENERO",
]

CATEGORICAL_COLUMNS = ["CANAL", "PRODUCTO", "CODIGO CLIENTE", "RUN CLIENTE"]

# Parámetros de procesamiento paralelo
CHUNK_SIZE = 100_000
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
TRAIN_TEST_RATIO = 0.7
