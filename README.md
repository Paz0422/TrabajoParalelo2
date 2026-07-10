# Análisis Estadístico de Ventas - Cruz Morada

Trabajo práctico de **Computación Paralela y Distribuida** (UTEM).  
Análisis estadístico avanzado con procesamiento paralelo sobre datos de ventas de farmacias.

## Requisitos

- Python 3.10+
- Archivo `ventas_completas.csv` (descargar desde [Google Drive](https://drive.google.com/file/d/15jLBlJ9eMQSoHsoCMnFWBGopr98FIHlK/view?usp=sharing))

## Instalación

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Coloca el CSV en la carpeta `data/`:

```
data/ventas_completas.csv
```

## Ejecución

El archivo CSV **debe pasarse por línea de comandos**:

```bash
# Análisis completo
python main.py --csv data/ventas_completas.csv

# Con Dask (archivos muy grandes)
python main.py --csv data/ventas_completas.csv --dask

# Prueba rápida con subset
python main.py --csv data/ventas_completas.csv --n-rows 50000

# Semilla de reproducibilidad
set CPYD_SEED=42
python main.py --csv data/ventas_completas.csv
```

### Etapas individuales

```bash
python main.py --csv data/ventas_completas.csv --stage preprocess
python main.py --csv data/ventas_completas.csv --stage eda
python main.py --csv data/ventas_completas.csv --stage inference
```

### Informe técnico (PDF)

Tras correr `main.py` sobre el dataset completo (genera `outputs/resultados.json` y los gráficos
en `outputs/eda/`), el informe técnico se regenera con:

```bash
pip install xhtml2pdf
python generate_report.py
```

Esto produce `Informe_Tecnico_CruzMorada.pdf` en la raíz del proyecto, con todas las cifras e
interpretaciones tomadas directamente de `outputs/resultados.json` (no hay números hardcodeados
salvo la narrativa).

## Estructura del proyecto

```
TrabajoParalelo2/
├── main.py                 # Punto de entrada CLI
├── requirements.txt
├── data/                   # CSV de entrada
├── outputs/                # Resultados (JSON, gráficos)
└── src/cruz_morada/
    ├── config.py           # Semilla CPYD_SEED, rutas, columnas
    ├── loader.py           # Carga por chunks / Dask
    ├── parallel/           # Procesamiento paralelo (multiprocessing)
    ├── preprocessing/      # Limpieza, outliers, variables derivadas
    ├── eda/                # Estadística descriptiva, visualizaciones, series temporales
    └── inference/          # Hipótesis, regresión, clustering
```

## Entregables del trabajo

| Criterio | Peso | Módulo |
|----------|------|--------|
| Preprocesamiento y limpieza | 20% | `preprocessing/` |
| Análisis exploratorio (EDA) | 30% | `eda/` |
| Inferencia y modelado | 30% | `inference/` |
| Rigor metodológico + informe PDF | 20% | Documentar en informe |

## Notas

- Variable de entorno `CPYD_SEED` controla la reproducibilidad (default: 42).
- Los resultados numéricos se guardan en `outputs/resultados.json`.
- Los gráficos EDA se guardan en `outputs/eda/`.
- El CSV real usa `;` como separador (no coma) y campos entre comillas; `loader.py` lo detecta
  automáticamente. Las columnas `NOMBRES`/`APELLIDOS` (PII no usada en el análisis) se descartan
  en la carga para reducir memoria.
- `UNIDADES` es constante (=1) en el 100% de los registros reales (dataset a nivel de línea de
  producto). Se usa `ITEMS POR BOLETA` como proxy real de volumen de compra en la regresión y en
  la hipótesis H2.
- Hipótesis validadas en `inference/hypotheses.py`: H1 (APP vs WEB, ejemplo del enunciado), H2
  (descuento vs volumen, ejemplo adaptado), H3 (fin de semana vs día de semana), H4 (edad vs
  monto), H5 (género vs monto).
