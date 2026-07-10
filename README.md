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
python analizar_ventas.py --csv data/ventas_completas.csv

# Con Dask (archivos muy grandes)
python analizar_ventas.py --csv data/ventas_completas.csv --dask

# Prueba rápida con subset
python analizar_ventas.py --csv data/ventas_completas.csv --n-rows 50000

# Semilla de reproducibilidad
set CPYD_SEED=42
python analizar_ventas.py --csv data/ventas_completas.csv

# Limpieza y transformación también en paralelo (por defecto van secuenciales:
# ver "Notas" más abajo, resultó más lento en las mediciones)
python analizar_ventas.py --csv data/ventas_completas.csv --parallel-preprocess
```

### Etapas individuales

```bash
python analizar_ventas.py --csv data/ventas_completas.csv --stage preprocess
python analizar_ventas.py --csv data/ventas_completas.csv --stage eda
python analizar_ventas.py --csv data/ventas_completas.csv --stage inference
```

### Informe técnico (PDF)

Tras correr `analizar_ventas.py` sobre el dataset completo (genera `outputs/resultados.json` y los
gráficos en `outputs/eda/`), el informe técnico se regenera con:

```bash
pip install xhtml2pdf
python generar_informe.py
```

Esto produce `Informe_Tecnico_CruzMorada.pdf` en la raíz del proyecto, con todas las cifras e
interpretaciones tomadas directamente de `outputs/resultados.json` (no hay números hardcodeados
salvo la narrativa).

## Estructura del proyecto

```
TrabajoParalelo2/
├── analizar_ventas.py       # Punto de entrada CLI
├── generar_informe.py       # Genera el informe técnico en PDF
├── requirements.txt
├── data/                    # CSV de entrada
├── outputs/                 # Resultados (JSON, gráficos)
└── src/cruz_morada/
    ├── configuracion.py     # Semilla CPYD_SEED, rutas, columnas
    ├── carga_datos.py       # Carga por chunks / Dask
    ├── paralelo/            # Procesamiento paralelo (multiprocessing)
    ├── preprocesamiento/    # Limpieza, outliers, variables derivadas
    ├── analisis_exploratorio/  # Estadística descriptiva, visualizaciones, series temporales
    └── inferencia/          # Hipótesis, regresión, clustering
```

## Entregables del trabajo

| Criterio | Peso | Módulo |
|----------|------|--------|
| Preprocesamiento y limpieza | 20% | `preprocesamiento/` |
| Análisis exploratorio (EDA) | 30% | `analisis_exploratorio/` |
| Inferencia y modelado | 30% | `inferencia/` |
| Rigor metodológico + informe PDF | 20% | Documentar en informe |

## Notas

- Variable de entorno `CPYD_SEED` controla la reproducibilidad (default: 42).
- Los resultados numéricos se guardan en `outputs/resultados.json`.
- Los gráficos EDA se guardan en `outputs/eda/`.
- El CSV real usa `;` como separador (no coma) y campos entre comillas; `carga_datos.py` lo detecta
  automáticamente. Las columnas `NOMBRES`/`APELLIDOS` (PII no usada en el análisis) se descartan
  en la carga para reducir memoria.
- `UNIDADES` es constante (=1) en el 100% de los registros reales (dataset a nivel de línea de
  producto). Se usa `ITEMS POR BOLETA` como proxy real de volumen de compra en la regresión y en
  la hipótesis H2.
- Hipótesis validadas en `inferencia/hipotesis.py`: H1 (APP vs WEB, ejemplo del enunciado), H2
  (descuento vs volumen, ejemplo adaptado), H3 (fin de semana vs día de semana), H4 (edad vs
  monto), H5 (género vs monto).
- El cálculo de estadísticos por LOCAL siempre corre en paralelo. Limpieza y transformación de
  variables también están implementadas en paralelo (`--parallel-preprocess`, particionando por
  hash de `CODIGO CLIENTE` para que `FRECUENCIA COMPRA`/`ITEMS POR BOLETA` se calculen correctamente),
  pero **no van en paralelo por defecto**: medido sobre el dataset completo, la versión paralela
  (2,4x más lenta) pierde contra pandas vectorizado en un solo proceso, porque el costo de
  serializar particiones grandes entre procesos supera la ganancia de usar más núcleos. Ver
  sección 1 del informe para el detalle.
