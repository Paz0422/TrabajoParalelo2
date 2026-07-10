#!/usr/bin/env python3
"""
Genera el informe técnico (PDF) a partir de outputs/resultados.json y los
gráficos de outputs/eda/. Ejecutar después de correr analizar_ventas.py sobre
el dataset completo:

    python generar_informe.py

Requiere xhtml2pdf (pip install xhtml2pdf).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
EDA_DIR = OUTPUTS / "eda"
RESULTS_PATH = OUTPUTS / "resultados.json"
REPORT_PATH = ROOT / "Informe_Tecnico_CruzMorada.pdf"


def img_b64(name: str) -> str:
    data = (EDA_DIR / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def fmt(n, decimals=2):
    """Formatea números grandes con separador de miles (estilo chileno: punto)."""
    if n is None:
        return "-"
    if isinstance(n, float) and abs(n) < 1 and n != 0:
        return f"{n:.{max(decimals,4)}f}"
    s = f"{n:,.{decimals}f}"
    # estilo latam: punto para miles, coma para decimales
    integer, _, dec = s.partition(".")
    integer = integer.replace(",", ".")
    return f"{integer},{dec}" if decimals else integer


def pval_fmt(p):
    if p is None:
        return "-"
    if p < 0.0001:
        return "p < 0,0001"
    return f"p = {p:.4f}".replace(".", ",")


def sig_txt(sig: bool) -> str:
    return "significativo, p < 0,05" if sig else "no significativo, p ≥ 0,05"


def main() -> None:
    r = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    prep = r["preprocessing"]
    eda = r["eda"]
    inf = r["inference"]
    ht = inf["hypothesis_tests"]
    reg = inf["regression"]
    clu = inf["clustering"]
    desc = eda["descriptive_stats"]

    n_raw = r["n_rows"]
    n_clean = int(desc["count"]["MONTO APLICADO"])
    n_removed = n_raw - n_clean

    css = """
    @page { size: A4; margin: 2.8cm 2.5cm 2.8cm 2.5cm; }
    body { font-family: 'Times New Roman', Georgia, serif; font-size: 11.5pt; color: #111111; line-height: 1.5; }
    h1 { font-size: 15pt; text-align: center; margin-bottom: 2px; font-weight: bold; }
    h2 { font-size: 13pt; margin-top: 22px; margin-bottom: 8px; border-bottom: 1px solid #999; padding-bottom: 2px; }
    h3 { font-size: 12pt; margin-top: 16px; margin-bottom: 4px; font-style: italic; }
    h4 { font-size: 11.5pt; margin-top: 12px; margin-bottom: 3px; font-weight: bold; }
    p { margin: 6px 0; text-align: justify; }
    .center { text-align: center; }
    .subtitle { font-size: 11.5pt; text-align: center; margin-top: 2px; }
    .meta { font-size: 10.5pt; text-align: center; margin-top: 16px; margin-bottom: 6px; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; font-size: 10pt; }
    th { border-bottom: 1.2px solid #111; border-top: 1.2px solid #111; padding: 4px 6px; text-align: left; font-weight: bold; }
    td { padding: 3px 6px; border-bottom: 0.5px solid #ccc; }
    table.ultima tr:last-child td { border-bottom: 1.2px solid #111; }
    .nota { margin: 6px 0 10px 18px; font-size: 11pt; }
    img.plot { width: 420px; margin: 10px auto 2px auto; display: block; border: 0.5px solid #999; }
    .caption { font-size: 9.5pt; color: #333; text-align: center; margin-bottom: 12px; font-style: italic; }
    .footer-note { font-size: 9pt; color: #555; margin-top: 30px; border-top: 0.5px solid #999; padding-top: 6px; }
    code { font-family: 'Courier New', monospace; font-size: 10pt; }
    """

    html = f"""
    <html><head><meta charset="utf-8"><style>{css}</style></head>
    <body>

    <div class="meta">
        Universidad Tecnológica Metropolitana<br/>
        Departamento de Computación e Informática<br/>
        Computación Paralela y Distribuida
    </div>

    <h1>Análisis Estadístico de Datos de Ventas — Cruz Morada</h1>
    <div class="subtitle">Inferencia y Modelado con procesamiento paralelo</div>
    <div class="meta">
        Profesor: Sebastián Salazar Molina — Entrega: 10 de julio de 2026<br/>
        Semilla de reproducibilidad (CPYD_SEED): {r['seed']}
    </div>

    <h2>Resumen</h2>
    <p>
    Se procesaron {fmt(n_raw,0)} transacciones de venta de la cadena Cruz Morada (noviembre 2023
    a diciembre 2024, 13 meses, 4 canales de venta y más de 700 locales), aplicando limpieza,
    detección de outliers, ingeniería de variables y procesamiento paralelo por particiones. Tras
    la limpieza quedaron {fmt(n_clean,0)} registros válidos ({fmt(n_removed,0)} descartados por
    errores de registro, ver sección 2.3). Se validaron 5 hipótesis estadísticas, un modelo de
    regresión lineal (R² ajustado = {fmt(reg['r_squared_adj'],3)}) y un modelo de clustering
    K-Means (silhouette = {fmt(clu['silhouette_score'],3)}) sobre la base de clientes.
    </p>

    <h2>1. Descripción de la solución implementada</h2>
    <p>
    La solución es una aplicación en Python (<code>analizar_ventas.py</code>) organizada en el
    paquete <code>src/cruz_morada</code>, con módulos separados para la carga de datos
    (<code>carga_datos.py</code>), procesamiento paralelo (<code>paralelo/</code>), preprocesamiento
    (<code>preprocesamiento/</code>), análisis exploratorio (<code>analisis_exploratorio/</code>) e
    inferencia/modelado (<code>inferencia/</code>). El archivo CSV se recibe por línea de comandos
    (<code>--csv data/ventas_completas.csv.gz</code>) y puede procesarse por fragmentos (chunking,
    tamaño configurable) o con Dask para lectura diferida (<code>--dask</code>), según el volumen
    de datos.
    </p>
    <p>
    El enunciado pide paralelizar las etapas de limpieza, transformación y cálculo de estadísticos.
    Las tres están implementadas mediante <code>concurrent.futures.ProcessPoolExecutor</code>
    (módulo <code>paralelo/procesador.py</code>), que reparte particiones del DataFrame entre varios
    procesos worker (por defecto <code>cpu_count() - 1</code>, ajustable con <code>--workers</code>)
    y mantiene un único pool de procesos vivo durante todo el pipeline, en vez de crear y destruir
    workers en cada etapa. El cálculo de estadísticos agregados por local (conteo y suma de MONTO
    APLICADO) siempre corre en paralelo, particionando por <code>LOCAL</code>.
    </p>
    <p>
    La limpieza y la transformación de variables también están implementadas en paralelo
    (<code>--parallel-preprocess</code>), pero <b>no se usan en paralelo por defecto</b>, y esto
    merece explicarse porque no es una omisión sino una decisión basada en medir, no en suponer.
    Dos de las variables derivadas (FRECUENCIA COMPRA e ITEMS POR BOLETA) calculan
    <code>groupby(CODIGO CLIENTE)</code> y <code>groupby(BOLETA)</code> internamente; particionar
    de forma ingenua (por ejemplo por LOCAL o por fecha, como sugiere el enunciado) puede repartir
    las transacciones de un mismo cliente entre distintos procesos y subestimar esas agregaciones.
    Por eso, cuando se activa el modo paralelo, la transformación particiona por un hash
    determinista de CODIGO CLIENTE: como una boleta pertenece siempre a un único cliente, esto
    garantiza que ningún cliente ni boleta quede repartido entre particiones. La mediana de EDAD
    usada para imputar valores implausibles también se calcula una sola vez sobre el DataFrame
    completo y se pasa como constante a cada partición, para no depender de cómo se particionen los
    datos.
    </p>
    <p>
    Habiendo resuelto la correctitud, medimos el desempeño real sobre las {fmt(n_raw,0)} filas del
    dataset completo, y el resultado fue contraintuitivo: la versión secuencial (limpieza +
    transformación con pandas vectorizado, un solo proceso) tomó 2,84 s + 4,14 s ≈ 7,0 s, mientras
    que la versión paralela (11 workers) tomó 9,99 s + 7,07 s ≈ 17,1 s — más de 2 veces más lenta.
    La razón es que <code>clean_data</code> y <code>create_derived_features</code> ya son
    operaciones vectorizadas de pandas (esencialmente bucles en C sobre columnas completas);
    partirlas en particiones y enviarlas a procesos separados agrega el costo de serializar
    (pickle) varios cientos de miles de filas por partición y transferirlas entre procesos, un costo
    que en este caso supera la ganancia de usar más núcleos de CPU. Por eso el pipeline usa
    procesamiento secuencial vectorizado por defecto para estas dos etapas, y dispone del camino
    paralelo como una implementación correcta y disponible (<code>--parallel-preprocess</code>),
    documentada y medida, para volúmenes de datos donde el trabajo por partición sea lo
    suficientemente pesado como para justificar el costo de comunicación entre procesos.
    </p>
    <p>
    Sobre la decisión de manejo de memoria: con {fmt(n_raw,0)} filas (~217 MB comprimidos), el
    DataFrame completo cabe en memoria una vez que se descartan las columnas de identificación
    personal que no se usan en el análisis (<code>NOMBRES</code>, <code>APELLIDOS</code>), por lo
    que se optó por lectura en chunks que se concatenan en un único DataFrame en memoria, en vez de
    mantener todo el pipeline fuera de memoria ("out-of-core"). Esto simplifica bastante las
    operaciones que necesitan ver todos los datos a la vez (correlaciones, regresión, clustering),
    que de otra forma requerirían un motor distribuido completo. Para volúmenes que ya no quepan en
    RAM, el flag <code>--dask</code> activa lectura lazy con Dask sin cambiar el resto del pipeline.
    Esta decisión se documenta explícitamente porque durante el desarrollo la máquina disponible
    tenía recursos limitados (16 GB RAM, ~2,5 GB libres al momento de correr el pipeline completo),
    lo que fue una restricción real (ver sección 6).
    </p>
    <p>
    Toda operación con aleatoriedad (imputación, partición train/test 70/30, inicialización de
    K-Means) usa la semilla fijada por la variable de entorno <code>CPYD_SEED</code> (valor por
    defecto y usado en esta ejecución: {r['seed']}).
    </p>

    <h2>2. Datos y preprocesamiento</h2>

    <h3>2.1 Carga y consolidación</h3>
    <p>
    El enunciado describe el CSV con columnas separadas por coma, pero el archivo real
    (<code>ventas_completas.csv.gz</code>) usa punto y coma (;) como separador y todos los campos
    entre comillas dobles. El loader detecta automáticamente el separador correcto revisando la
    primera línea del archivo antes de leerlo con pandas, para que el pipeline no falle
    interpretando cada fila como una sola columna. Al ser un único archivo consolidado no fue
    necesario unificar múltiples fuentes.
    </p>

    <h3>2.2 Valores faltantes</h3>
    <p>
    Se implementó un test aproximado de MCAR (correlación entre indicadores de ausencia de cada
    columna numérica) y un reporte de nulos por columna. Sobre los {fmt(n_raw,0)} registros (y
    verificado también con un barrido directo del archivo en busca de campos vacíos o tokens como
    NULL/NA) no se encontraron valores faltantes explícitos en ninguna columna. Esto difiere del
    ejemplo del enunciado, que sugería nulos en <code>PORCENTAJE DESCUENTO</code> o
    <code>FECHA NACIMIENTO</code>, pero es lo que muestra el archivo entregado. El pipeline
    igualmente conserva la lógica de imputación (mediana para fechas, 0 para descuento, moda para
    género) por si en algún momento se usa con datos que sí tengan nulos, y el test de MCAR se
    ejecuta y reporta de todas formas.
    </p>

    <h3>2.3 Errores de registro detectados</h3>
    <p>
    Se identificaron tres tipos de inconsistencias, distintas a "valores faltantes" pero igual de
    relevantes para la calidad del análisis:
    </p>
    <table class="ultima">
        <tr><th>Problema detectado</th><th>Registros afectados</th><th>Tratamiento aplicado</th></tr>
        <tr><td>BOLETA = 0 (no corresponde a un documento tributario real)</td>
            <td>8 de {fmt(n_raw,0)} (&lt;0,001%)</td><td>Eliminados</td></tr>
        <tr><td>PORCENTAJE DESCUENTO fuera de rango [0,1] (se encontró un registro en 1,17, es decir 117%)</td>
            <td>1 registro</td><td>Eliminado</td></tr>
        <tr><td>EDAD implausible (negativa o mayor a 100 años), originada en FECHA NACIMIENTO corrupta</td>
            <td>3.323 (0,10%)</td><td>Invalidada e imputada con la mediana de edades plausibles (48,6 años)</td></tr>
    </table>
    <p>
    Estos casos se interpretan como errores de registro y no como casos de negocio reales: una
    boleta 0 no existe, un descuento sobre el 100% es imposible, y una edad de -5.944 o 825 años
    (valores observados antes del tratamiento) no corresponde a ningún cliente real. El umbral [0,
    100] años para EDAD se dejó a propósito conservador y queda documentado en el código
    (<code>variables_derivadas.py</code>, constantes <code>MIN_EDAD_PLAUSIBLE</code> y
    <code>MAX_EDAD_PLAUSIBLE</code>).
    </p>

    <h3>2.4 Detección de outliers (variables continuas)</h3>
    <p>
    Se aplicaron los métodos IQR (rango intercuartílico, factor 1,5) y Z-score (umbral 3) sobre
    MONTO APLICADO:
    </p>
    <table class="ultima">
        <tr><th>Método</th><th>Outliers detectados</th><th>% del total</th></tr>
        <tr><td>IQR (1,5×RIC)</td><td>{fmt(prep['outliers_monto']['n_iqr_outliers'],0)}</td><td>{fmt(prep['outliers_monto']['pct_iqr_outliers'])}%</td></tr>
        <tr><td>Z-score (|z|&gt;3)</td><td>{fmt(prep['outliers_monto']['n_zscore_outliers'],0)}</td><td>{fmt(prep['outliers_monto']['pct_zscore_outliers'])}%</td></tr>
    </table>
    <p>
    A diferencia de los errores de la sección 2.3, estos no se eliminan. MONTO APLICADO tiene una
    distribución fuertemente asimétrica a la derecha (asimetría =
    {fmt(desc['skewness']['MONTO APLICADO'])}, curtosis = {fmt(desc['kurtosis']['MONTO APLICADO'])}),
    y montos altos (hasta ${fmt(desc['max']['MONTO APLICADO'],0)} CLP) son consistentes con compras
    de medicamentos de alto costo (tratamientos crónicos, oncológicos, etc.), no con errores de
    digitación. Se interpretan entonces como casos de negocio reales de cola larga y se conservan
    en el análisis. UNIDADES no presenta outliers (0% por ambos métodos) por la razón que se
    explica a continuación.
    </p>

    <p>
    <b>Hallazgo relevante.</b> UNIDADES es constante e igual a 1 en el 100% de los
    {fmt(n_raw,0)} registros. Esto revela que el dataset está a nivel de línea de producto por
    boleta (una fila = un producto distinto comprado), no a nivel de cantidad total. Por lo tanto
    UNIDADES no aporta varianza para ningún análisis estadístico (correlaciones, regresión, ANOVA),
    y se documenta así en vez de forzar pruebas degeneradas. Como proxy real del volumen de compra
    se creó la variable derivada ITEMS POR BOLETA (cantidad de líneas de producto distintas por
    número de boleta), que sí varía (media = {fmt(desc['mean']['ITEMS POR BOLETA'])}, máximo =
    {fmt(desc['max']['ITEMS POR BOLETA'],0)}) y se usa en su reemplazo en el modelo de regresión y
    en la hipótesis H2.
    </p>

    <h3>2.5 Variables derivadas</h3>
    <table class="ultima">
        <tr><th>Variable</th><th>Definición</th><th>Media</th><th>Nota</th></tr>
        <tr><td>MONTO POR UNIDAD</td><td>MONTO APLICADO / UNIDADES</td>
            <td>${fmt(desc['mean']['MONTO POR UNIDAD'],0)}</td>
            <td>Idéntica a MONTO APLICADO (UNIDADES=1 siempre); se mantiene por completitud del enunciado</td></tr>
        <tr><td>EDAD</td><td>(FECHA − FECHA NACIMIENTO) / 365,25, saneada</td>
            <td>{fmt(desc['mean']['EDAD'])} años</td><td>3.323 valores implausibles imputados (ver 2.3)</td></tr>
        <tr><td>FRECUENCIA COMPRA</td><td>N° de boletas distintas por CODIGO CLIENTE</td>
            <td>{fmt(desc['mean']['FRECUENCIA COMPRA'])}</td><td>Máximo observado: {fmt(desc['max']['FRECUENCIA COMPRA'],0)} boletas por cliente</td></tr>
        <tr><td>ITEMS POR BOLETA</td><td>N° de líneas de producto por BOLETA</td>
            <td>{fmt(desc['mean']['ITEMS POR BOLETA'])}</td><td>Proxy real de volumen (ver hallazgo de 2.4)</td></tr>
        <tr><td>HORA / DIA SEMANA / MES / ANIO / ES_FIN_DE_SEMANA</td>
            <td>Extraídas de FECHA</td><td>-</td><td>Usadas en análisis temporal e hipótesis H3</td></tr>
    </table>
    <p>
    No se aplicó estandarización a las variables usadas en regresión/ANOVA, para mantener la
    interpretación directa de los coeficientes en sus unidades originales; sí se estandarizan las
    variables de clientes antes del clustering (sección 4.3), documentando el scaler ajustado sobre
    el conjunto de entrenamiento.
    </p>

    <h2>3. Análisis exploratorio estadístico</h2>

    <h3>3.1 Estadística descriptiva</h3>
    <table class="ultima">
        <tr><th>Variable</th><th>Media</th><th>Desv. Est.</th><th>Mediana</th><th>Asimetría</th><th>Curtosis</th></tr>
        <tr><td>MONTO APLICADO (CLP)</td><td>{fmt(desc['mean']['MONTO APLICADO'],0)}</td><td>{fmt(desc['std']['MONTO APLICADO'],0)}</td><td>{fmt(desc['50%']['MONTO APLICADO'],0)}</td><td>{fmt(desc['skewness']['MONTO APLICADO'])}</td><td>{fmt(desc['kurtosis']['MONTO APLICADO'])}</td></tr>
        <tr><td>PORCENTAJE DESCUENTO</td><td>{fmt(desc['mean']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['std']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['50%']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['skewness']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['kurtosis']['PORCENTAJE DESCUENTO'])}</td></tr>
        <tr><td>EDAD (años)</td><td>{fmt(desc['mean']['EDAD'])}</td><td>{fmt(desc['std']['EDAD'])}</td><td>{fmt(desc['50%']['EDAD'])}</td><td>{fmt(desc['skewness']['EDAD'])}</td><td>{fmt(desc['kurtosis']['EDAD'])}</td></tr>
        <tr><td>ITEMS POR BOLETA</td><td>{fmt(desc['mean']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['std']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['50%']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['skewness']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['kurtosis']['ITEMS POR BOLETA'])}</td></tr>
    </table>
    <p>
    El ticket promedio es de ${fmt(desc['mean']['MONTO APLICADO'],0)} CLP, pero la mediana
    (${fmt(desc['50%']['MONTO APLICADO'],0)}) es notablemente menor que la media, lo que indica una
    distribución con cola larga a la derecha: la mayoría de las compras son de monto moderado, pero
    hay compras puntuales muy altas que "tiran" el promedio hacia arriba (la curtosis de
    {fmt(desc['kurtosis']['MONTO APLICADO'])} muestra una concentración de valores extremos muy
    superior a una distribución normal). La edad promedio de los clientes es de
    {fmt(desc['mean']['EDAD'])} años, con alta dispersión (±{fmt(desc['std']['EDAD'])} años),
    reflejando una base de clientes heterogénea, típica de una farmacia.
    </p>

    <h4>Test de normalidad (Shapiro-Wilk y Kolmogorov-Smirnov)</h4>
    <table class="ultima">
        <tr><th>Variable</th><th>Shapiro-Wilk (p)</th><th>Kolmogorov-Smirnov (p)</th><th>¿Normal?</th></tr>
        <tr><td>MONTO APLICADO</td><td>{pval_fmt(eda['normality_tests']['MONTO APLICADO']['shapiro']['p_value'])}</td>
            <td>{pval_fmt(eda['normality_tests']['MONTO APLICADO']['kolmogorov_smirnov']['p_value'])}</td><td>No</td></tr>
        <tr><td>PORCENTAJE DESCUENTO</td><td>{pval_fmt(eda['normality_tests']['PORCENTAJE DESCUENTO']['shapiro']['p_value'])}</td>
            <td>{pval_fmt(eda['normality_tests']['PORCENTAJE DESCUENTO']['kolmogorov_smirnov']['p_value'])}</td><td>No</td></tr>
    </table>
    <p>
    Ambas pruebas rechazan la normalidad (p &lt; 0,0001) para las dos variables continuas
    principales. Por eso el pipeline selecciona automáticamente correlación de Spearman (basada en
    rangos, no asume normalidad) en vez de Pearson para la matriz de correlación general, y cada
    prueba de hipótesis de comparación de medias verifica normalidad por grupo antes de elegir
    entre t-test/Welch o Mann-Whitney U.
    </p>

    <img class="plot" src="{img_b64('hist_monto_aplicado.png')}"/>
    <div class="caption">Figura 1. Distribución de MONTO APLICADO: fuerte asimetría a la derecha, consistente con el rechazo de normalidad.</div>

    <img class="plot" src="{img_b64('hist_porcentaje_descuento.png')}"/>
    <div class="caption">Figura 2. Distribución de PORCENTAJE DESCUENTO: se concentra en tramos discretos (≈20%, 35%, 40%, 50%) en vez de ser continua, lo que sugiere tramos de convenio o copago típicos de farmacias (Fonasa/Isapre) en vez de descuentos arbitrarios.</div>

    <h3>3.2 Boxplot por categoría</h3>
    <img class="plot" src="{img_b64('boxplot_monto_por_canal.png')}"/>
    <div class="caption">Figura 3. MONTO APLICADO por CANAL. Las medianas son similares entre canales, pero POS y WEB concentran más valores extremos (mayor volumen de transacciones).</div>

    <h3>3.3 Matriz de correlación (Spearman) con significancia</h3>
    <img class="plot" src="{img_b64('correlation_matrix.png')}"/>
    <div class="caption">Figura 4. UNIDADES aparece en blanco por ser constante (correlación no definida). MONTO APLICADO y PORCENTAJE DESCUENTO muestran correlación positiva moderada.</div>
    <p>
    La correlación de Spearman entre MONTO APLICADO y PORCENTAJE DESCUENTO es de
    {fmt(eda['correlation']['MONTO APLICADO']['PORCENTAJE DESCUENTO'])}
    ({pval_fmt(eda['correlation_pvalues']['MONTO APLICADO']['PORCENTAJE DESCUENTO'])}), es decir,
    estadísticamente significativa: a mayor monto de compra, mayor tiende a ser el porcentaje de
    descuento aplicado. Esto es coherente con una lectura de negocio: los tratamientos de mayor
    valor (crónicos, especialidades) suelen estar cubiertos por convenios con mayor cobertura
    porcentual.
    </p>

    <h3>3.4 Asociación entre variables categóricas</h3>
    <table class="ultima">
        <tr><th>Prueba</th><th>Estadístico</th><th>gl</th><th>p-value</th><th>Conclusión</th></tr>
        <tr><td>Chi-cuadrado: CANAL × LOCAL</td><td>χ² = {fmt(eda['chi2_canal_local']['chi2'],0)}</td>
            <td>{eda['chi2_canal_local']['dof']}</td><td>{pval_fmt(eda['chi2_canal_local']['p_value'])}</td>
            <td>{sig_txt(eda['chi2_canal_local']['significant_005'])}</td></tr>
        <tr><td>ANOVA: MONTO APLICADO ~ CANAL</td><td>F = {fmt(eda['anova_monto_canal']['f_statistic'])}</td>
            <td>-</td><td>{pval_fmt(eda['anova_monto_canal']['p_value'])}</td>
            <td>{sig_txt(eda['anova_monto_canal']['significant_005'])}</td></tr>
    </table>
    <p>
    El canal de venta (POS, WEB, APP, CCT) y el local donde se realiza la compra no son
    independientes (p &lt; 0,05): la mezcla de canales varía significativamente entre locales
    (algunos locales concentran más ventas WEB/APP, lo cual es esperable si corresponden a zonas
    con distinto perfil de clientes). El monto promedio de compra también difiere
    significativamente entre canales (ANOVA, p &lt; 0,05), aunque como se ve en el boxplot (Figura
    3) las medianas son parecidas y la diferencia la impulsa principalmente la cola de valores
    altos.
    </p>

    <h3>3.5 Patrones temporales</h3>
    <p>
    Se agregaron las ventas diarias (suma de MONTO APLICADO por día) sobre el período completo
    (noviembre 2023 a diciembre 2024) y se aplicó descomposición aditiva (tendencia + estacionalidad
    + residuo, período = 7 días) y análisis de autocorrelación.
    </p>
    <img class="plot" src="{img_b64('seasonal_decomposition.png')}"/>
    <div class="caption">Figura 5. Descomposición de ventas diarias. La venta diaria promedio es de
    ${fmt(eda['daily_sales_summary']['mean'],0)} CLP (±${fmt(eda['daily_sales_summary']['std'],0)}).
    Se observa una tendencia creciente sostenida durante el período y un fuerte componente
    estacional semanal superpuesto (picos y valles que se repiten cada 7 días). El día de mayor
    venta agregada fue el {eda['daily_sales_summary']['max_date'][:10]}.</div>

    <img class="plot" src="{img_b64('acf_pacf.png')}"/>
    <div class="caption">Figura 6. ACF y PACF de la serie diaria. La ACF muestra picos marcados y
    persistentes en los rezagos múltiplos de 7 (7, 14, 21, 28, 35), confirmando una estacionalidad
    semanal clara: el nivel de ventas de un día depende fuertemente del nivel del mismo día de la
    semana anterior. La PACF muestra que, controlando por rezagos intermedios, el rezago 7 sigue
    siendo el más relevante después del rezago 1.</div>

    <h2>4. Inferencia estadística</h2>

    <h3>4.1 Pruebas de hipótesis</h3>
    <p>Se validaron 2 hipótesis del enunciado (H1, H2) y 3 hipótesis propias (H3, H4, H5).</p>

    <h4>H1 — El ticket promedio en APP es mayor que en WEB (ejemplo del enunciado)</h4>
    <p>
    Test: {ht['H1_ticket_app_vs_web']['test']}. Media APP = ${fmt(ht['H1_ticket_app_vs_web']['mean_a'],0)},
    media WEB = ${fmt(ht['H1_ticket_app_vs_web']['mean_b'],0)} ({pval_fmt(ht['H1_ticket_app_vs_web']['p_value'])},
    {sig_txt(ht['H1_ticket_app_vs_web']['significant_005'])}).
    Aunque el ticket promedio observado en WEB es numéricamente mayor que en APP, la prueba de
    Mann-Whitney U indica que esta diferencia no es estadísticamente significativa: no hay
    evidencia suficiente para afirmar que el canal APP o WEB tenga, por sí solo, un ticket promedio
    distinto. Es un hallazgo que contradice la premisa del ejemplo del enunciado, y se reporta así
    porque es lo que muestran los datos reales.
    </p>

    <h4>H2 — El % de descuento afecta el volumen de compra (ejemplo del enunciado, adaptado)</h4>
    <p>
    Como se explicó en 2.4, UNIDADES es constante y no permite esta prueba tal como está planteada
    en el enunciado, así que se usa ITEMS POR BOLETA como variable dependiente. Test: regresión
    lineal simple. Pendiente = {fmt(ht['H2_descuento_vs_volumen']['slope'],4)}, R² =
    {fmt(ht['H2_descuento_vs_volumen']['r_squared'],6)} ({pval_fmt(ht['H2_descuento_vs_volumen']['p_value'])},
    {sig_txt(ht['H2_descuento_vs_volumen']['significant_005'])}).
    La relación es estadísticamente significativa (con más de 3 millones de registros, hasta
    efectos diminutos resultan significativos), pero un R² de
    {fmt(ht['H2_descuento_vs_volumen']['r_squared'],6)} indica que el porcentaje de descuento
    explica una fracción prácticamente nula de la variación en ítems por boleta. En términos de
    negocio, el descuento aplicado no es un predictor relevante del volumen de productos que lleva
    un cliente en su boleta, aunque la relación exista formalmente.
    </p>

    <h4>H3 (propia) — El ticket promedio difiere entre fin de semana y día de semana</h4>
    <p>
    Test: {ht['H3_monto_finde_vs_semana']['test']}. Media fin de semana = ${fmt(ht['H3_monto_finde_vs_semana']['mean_finde'],0)},
    media día de semana = ${fmt(ht['H3_monto_finde_vs_semana']['mean_semana'],0)} ({pval_fmt(ht['H3_monto_finde_vs_semana']['p_value'])},
    {sig_txt(ht['H3_monto_finde_vs_semana']['significant_005'])}).
    El ticket promedio en días de semana es significativamente mayor que en fines de semana. Una
    lectura de negocio plausible: las compras de mayor valor (tratamientos crónicos, retiro de
    recetas médicas) se concentran en días hábiles, mientras que el fin de semana concentra compras
    más pequeñas y espontáneas (por ejemplo analgésicos o cuidado personal).
    </p>

    <h4>H4 (propia) — La edad del cliente se correlaciona con el monto de compra</h4>
    <p>
    Test: Pearson, aplicado sobre EDAD (cuya distribución es más cercana a la normal que MONTO
    APLICADO). r = {fmt(ht['H4_edad_vs_monto']['correlation'])} ({pval_fmt(ht['H4_edad_vs_monto']['p_value'])},
    {sig_txt(ht['H4_edad_vs_monto']['significant_005'])}).
    Existe una correlación positiva pero muy débil entre la edad del cliente y el monto de compra:
    los clientes de mayor edad tienden, en promedio, a gastar levemente más, pero la edad por sí
    sola explica muy poco de la variación del monto (consistente con clientes de mayor edad
    asociados a tratamientos crónicos de mayor costo, aunque el efecto es marginal frente a otros
    factores, como el producto específico comprado).
    </p>

    <h4>H5 (propia) — El género del cliente influye en el monto promedio de compra</h4>
    <p>
    Test: ANOVA one-way. F = {fmt(ht['H5_genero_vs_monto']['f_statistic'])} ({pval_fmt(ht['H5_genero_vs_monto']['p_value'])},
    {sig_txt(ht['H5_genero_vs_monto']['significant_005'])}).
    Existe una diferencia estadísticamente significativa en el monto promedio de compra entre
    géneros. El tamaño de la muestra (más de 3 millones de registros) hace que incluso diferencias
    moderadas resulten altamente significativas, por lo que conviene revisar la magnitud absoluta
    de la diferencia antes de sacar conclusiones de negocio basadas solo en el p-value.
    </p>

    <h3>4.2 Modelado predictivo — Opción A: regresión lineal</h3>
    <p>
    Se modeló MONTO APLICADO ~ CANAL + LOCAL + ITEMS POR BOLETA + PORCENTAJE DESCUENTO mediante OLS
    (statsmodels), con partición train/test 70/30 (semilla {r['seed']}). UNIDADES se sustituyó por
    ITEMS POR BOLETA por la razón explicada en 2.4: incluir una columna constante en la regresión
    produce colinealidad perfecta con el intercepto (matriz de diseño no invertible), lo que se
    detectó en una corrida preliminar (coeficientes idénticos entre intercepto y UNIDADES) y se
    corrigió antes de la entrega.
    </p>
    <table class="ultima">
        <tr><th>Métrica</th><th>Valor</th></tr>
        <tr><td>R² ajustado</td><td>{fmt(reg['r_squared_adj'],4)}</td></tr>
        <tr><td>RMSE (test)</td><td>${fmt(reg['rmse_test'],0)}</td></tr>
        <tr><td>MAE (test)</td><td>${fmt(reg['mae_test'],0)}</td></tr>
    </table>
    <h4>Coeficientes y significancia</h4>
    <table class="ultima">
        <tr><th>Variable</th><th>Coeficiente</th><th>p-value</th></tr>
        <tr><td>Intercepto (canal APP, referencia)</td><td>{fmt(reg['coefficients']['Intercept'],1)}</td><td>{pval_fmt(reg['pvalues']['Intercept'])}</td></tr>
        <tr><td>CANAL = CCT</td><td>{fmt(reg['coefficients']['C(CANAL)[T.CCT]'],1)}</td><td>{pval_fmt(reg['pvalues']['C(CANAL)[T.CCT]'])}</td></tr>
        <tr><td>CANAL = POS</td><td>{fmt(reg['coefficients']['C(CANAL)[T.POS]'],1)}</td><td>{pval_fmt(reg['pvalues']['C(CANAL)[T.POS]'])}</td></tr>
        <tr><td>CANAL = WEB</td><td>{fmt(reg['coefficients']['C(CANAL)[T.WEB]'],1)}</td><td>{pval_fmt(reg['pvalues']['C(CANAL)[T.WEB]'])}</td></tr>
        <tr><td>LOCAL</td><td>{fmt(reg['coefficients']['LOCAL'],4)}</td><td>{pval_fmt(reg['pvalues']['LOCAL'])}</td></tr>
        <tr><td>ITEMS POR BOLETA</td><td>{fmt(reg['coefficients']["Q('ITEMS POR BOLETA')"],2)}</td><td>{pval_fmt(reg['pvalues']["Q('ITEMS POR BOLETA')"])}</td></tr>
        <tr><td>PORCENTAJE DESCUENTO</td><td>{fmt(reg['coefficients']["Q('PORCENTAJE DESCUENTO')"],1)}</td><td>{pval_fmt(reg['pvalues']["Q('PORCENTAJE DESCUENTO')"])}</td></tr>
    </table>

    <h4>Diagnóstico de supuestos y multicolinealidad (VIF)</h4>
    <table class="ultima">
        <tr><th>Variable</th><th>VIF</th><th>Lectura</th></tr>
        <tr><td>LOCAL</td><td>{fmt(reg['vif']['LOCAL'])}</td><td>Sin problema (VIF &lt; 5)</td></tr>
        <tr><td>ITEMS POR BOLETA</td><td>{fmt(reg['vif']['ITEMS POR BOLETA'])}</td><td>Sin problema (VIF &lt; 5)</td></tr>
        <tr><td>PORCENTAJE DESCUENTO</td><td>{fmt(reg['vif']['PORCENTAJE DESCUENTO'])}</td><td>Sin problema (VIF &lt; 5)</td></tr>
    </table>
    <p>
    Ningún VIF supera 5, por lo que no hay evidencia de multicolinealidad severa entre los
    predictores retenidos.
    </p>

    <h4>Diagnóstico de linealidad, homocedasticidad y normalidad de residuales</h4>
    <img class="plot" src="{img_b64('regresion_residuos_vs_ajustados.png')}"/>
    <div class="caption">Figura 7. Residuos vs. valores ajustados (muestra de 5.000 puntos del
    conjunto de entrenamiento). El patrón en forma de abanico, con la dispersión de los residuos
    creciendo junto con el valor ajustado, es evidencia visual de heterocedasticidad, no de una
    nube homogénea sin forma como asumiría un modelo bien especificado.</div>

    <img class="plot" src="{img_b64('regresion_qqplot_residuales.png')}"/>
    <div class="caption">Figura 8. Q-Q plot de los residuales. Los puntos se apartan claramente de
    la recta de referencia en ambas colas, sobre todo en la cola superior, mostrando una
    distribución de residuales de cola pesada y asimétrica, no normal.</div>

    <table class="ultima">
        <tr><th>Test</th><th>Supuesto evaluado</th><th>Estadístico</th><th>p-value</th><th>Conclusión</th></tr>
        <tr><td>Breusch-Pagan</td><td>Homocedasticidad (H0: varianza constante)</td>
            <td>{fmt(reg['diagnostics']['breusch_pagan']['statistic'],1)}</td>
            <td>{pval_fmt(reg['diagnostics']['breusch_pagan']['p_value'])}</td>
            <td>Se rechaza H0: heterocedástico</td></tr>
        <tr><td>Jarque-Bera</td><td>Normalidad de residuales (H0: distribución normal)</td>
            <td>{fmt(reg['diagnostics']['jarque_bera']['statistic'],1)}</td>
            <td>{pval_fmt(reg['diagnostics']['jarque_bera']['p_value'])}</td>
            <td>Se rechaza H0: no normales</td></tr>
    </table>
    <p>
    Los tres supuestos clásicos de OLS quedan comprometidos en algún grado. Sobre linealidad, el
    gráfico de residuos vs. ajustados (Figura 7) no muestra una curvatura sistemática marcada, pero
    sí una asimetría hacia residuos positivos grandes, coherente con la cola larga de MONTO
    APLICADO. El test de Breusch-Pagan rechaza la hipótesis de varianza constante
    ({pval_fmt(reg['diagnostics']['breusch_pagan']['p_value'])}): el modelo es heterocedástico, es
    decir, el error de predicción crece junto con el monto de la transacción. El test de
    Jarque-Bera también rechaza la normalidad de los residuales
    ({pval_fmt(reg['diagnostics']['jarque_bera']['p_value'])}, asimetría =
    {fmt(reg['diagnostics']['jarque_bera']['skew'])}, curtosis =
    {fmt(reg['diagnostics']['jarque_bera']['kurtosis'])}), consistente con el Q-Q plot (Figura 8).
    </p>
    <p>
    En la práctica esto significa que los p-values e intervalos de confianza reportados para los
    coeficientes son aproximados y probablemente subestiman la incertidumbre real, sobre todo para
    transacciones de monto alto (donde el error es mayor). No invalida las conclusiones cualitativas
    del modelo (qué variables importan y en qué dirección), pero sí desaconseja usarlo para
    construir intervalos de predicción precisos sin corregir por heterocedasticidad (por ejemplo,
    con errores estándar robustos tipo HC3, o modelando log(MONTO APLICADO) en vez del monto en
    escala original). Esta limitación es consistente con el R² ajustado moderado
    ({fmt(reg['r_squared_adj'],3)}) y con la distribución fuertemente asimétrica de MONTO APLICADO
    documentada en la sección 3.1; se reporta explícitamente en vez de ocultarse.
    </p>
    <p>
    El modelo explica solo el {fmt(reg['r_squared_adj']*100,1)}% de la variación del monto de
    compra. El coeficiente de PORCENTAJE DESCUENTO (+{fmt(reg['coefficients']["Q('PORCENTAJE DESCUENTO')"],0)})
    es, con diferencia, el más influyente: las transacciones con mayor porcentaje de descuento
    tienden a tener un monto final más alto. Esto no debe leerse como "el descuento causa un mayor
    gasto", sino como reflejo de que los tratamientos de mayor valor (crónicos, especialidades
    médicas) suelen tener convenios con mayor cobertura porcentual — la causalidad probablemente va
    en la dirección opuesta, o es bidireccional. Los canales POS y WEB muestran tickets
    significativamente más altos que APP (canal de referencia); CCT no muestra diferencia
    significativa, posiblemente por su bajo volumen de transacciones. LOCAL, al ser un identificador
    numérico y no una variable de negocio continua, se interpreta con cautela (ver limitaciones).
    </p>

    <h3>4.3 Modelado descriptivo — Opción B: clustering de clientes (K-Means)</h3>
    <p>
    Se segmentaron los clientes (agrupados por CODIGO CLIENTE) usando 4 variables agregadas: monto
    total gastado, total de ítems comprados, número de transacciones distintas y descuento promedio
    recibido, estandarizadas con StandardScaler antes de aplicar K-Means (k=4, semilla {r['seed']},
    entrenado sobre partición 70% y evaluado sobre el 100% de los clientes).
    </p>
    <table class="ultima">
        <tr><th>Métrica</th><th>Valor</th></tr>
        <tr><td>Silhouette score</td><td>{fmt(clu['silhouette_score'],3)}</td></tr>
        <tr><td>Inercia (WCSS)</td><td>{fmt(clu['inertia'],0)}</td></tr>
    </table>
    <table class="ultima">
        <tr><th>Clúster</th><th>N° de clientes</th><th>% del total</th></tr>
        {"".join(f"<tr><td>{k}</td><td>{fmt(v,0)}</td><td>{fmt(100*v/sum(clu['cluster_sizes'].values()),1)}%</td></tr>" for k, v in clu['cluster_sizes'].items())}
    </table>
    <p>
    Un silhouette score de {fmt(clu['silhouette_score'],3)} (escala -1 a 1) indica una separación
    moderada-buena entre los 4 segmentos de clientes: los grupos son razonablemente distinguibles
    entre sí. El clúster más grande concentra aproximadamente el
    {fmt(100*max(clu['cluster_sizes'].values())/sum(clu['cluster_sizes'].values()),0)}% de los
    clientes (perfil estándar, bajo gasto/baja frecuencia), mientras que el clúster más pequeño
    (~{fmt(100*min(clu['cluster_sizes'].values())/sum(clu['cluster_sizes'].values()),1)}%)
    probablemente corresponde a clientes de alto valor (alta frecuencia y/o alto gasto acumulado),
    útil para campañas de fidelización dirigidas. Cabe notar que, dado que un porcentaje
    considerable de CODIGO CLIENTE distintos realiza muy pocas compras (mediana de FRECUENCIA
    COMPRA = {fmt(desc['50%']['FRECUENCIA COMPRA'],0)}), varios clústeres pueden estar dominados por
    clientes de compra única.
    </p>

    <h3>4.4 Validación de modelos y extrapolabilidad</h3>
    <p>
    Ambos modelos se evaluaron con partición train/test 70/30 fijada por semilla, siguiendo el
    enunciado. El error absoluto medio de la regresión (MAE = ${fmt(reg['mae_test'],0)}) representa
    aproximadamente el {fmt(100*reg['mae_test']/desc['mean']['MONTO APLICADO'],0)}% del ticket
    promedio, un margen de error considerable para usarlo como predictor puntual de negocio, aunque
    sigue siendo útil para explicar tendencias agregadas.
    </p>
    <p>¿Es el modelo extrapolable? Con algunos matices importantes:</p>
    <ul>
        <li>Ventana temporal limitada: el dataset cubre 13 meses (noviembre 2023 a diciembre 2024);
        la tendencia creciente observada (Figura 5) podría no sostenerse indefinidamente y el
        modelo no la captura explícitamente, ya que no se incluyó un término de tendencia temporal
        en la regresión.</li>
        <li>Variable de producto omitida: el predictor más obvio del monto de una transacción, qué
        producto o SKU se compró, no se incluyó en el modelo por su altísima cardinalidad (cientos
        de SKUs), lo que limita el R² alcanzable. Un modelo por categoría de producto probablemente
        mejoraría bastante el ajuste.</li>
        <li>LOCAL como variable numérica continua: es un identificador, no una magnitud, así que su
        coeficiente lineal no tiene una interpretación de negocio directa (un modelo con efectos
        fijos por local, inviable aquí por la cardinalidad mayor a 700, sería más apropiado).</li>
        <li>Sesgo de canal: CCT representa una fracción muy pequeña de las transacciones (canal
        minoritario), por lo que sus estimaciones son menos confiables que las de POS/WEB/APP.</li>
        <li>El clustering es sensible a la definición de "cliente": con una fracción relevante de
        compradores de una sola transacción, los clústeres reflejan en parte esa heterogeneidad
        estructural más que un comportamiento diferenciado real.</li>
    </ul>
    <p>
    En síntesis, los modelos son útiles como herramientas descriptivas y de apoyo a decisiones
    agregadas (qué canal o local revisar, qué segmento de clientes priorizar), pero no deberían
    usarse como predictor puntual de cuánto gastará un cliente específico sin incorporar la
    variable de producto y una ventana temporal más larga.
    </p>

    <h2>5. Justificación de las librerías utilizadas</h2>
    <table class="ultima">
        <tr><th>Librería</th><th>Uso en el proyecto</th></tr>
        <tr><td>pandas</td><td>Carga, limpieza y transformación tabular del dataset completo</td></tr>
        <tr><td>NumPy</td><td>Operaciones numéricas vectorizadas de soporte</td></tr>
        <tr><td>Dask</td><td>Lectura diferida (lazy) opcional para volúmenes que excedan la RAM disponible</td></tr>
        <tr><td>SciPy (stats)</td><td>Pruebas de normalidad (Shapiro-Wilk, KS), correlación (Pearson/Spearman), t-test, Mann-Whitney U, ANOVA, chi-cuadrado</td></tr>
        <tr><td>statsmodels</td><td>Regresión OLS con fórmulas tipo R, cálculo de VIF, descomposición estacional, ACF/PACF</td></tr>
        <tr><td>scikit-learn</td><td>K-Means, StandardScaler, train_test_split, métricas (RMSE, MAE, silhouette)</td></tr>
        <tr><td>matplotlib / seaborn</td><td>Visualizaciones estadísticas (histogramas, boxplots, heatmaps, series de tiempo)</td></tr>
        <tr><td>concurrent.futures (librería estándar)</td><td>Paralelización real de estadísticos agregados por partición, sin dependencias adicionales</td></tr>
    </table>

    <h2>6. Dificultades encontradas y cómo se resolvieron</h2>
    <table class="ultima">
        <tr><th>Dificultad</th><th>Cómo se detectó</th><th>Resolución</th></tr>
        <tr><td>El CSV real usa ; como separador y comillas, no coma como indica el enunciado</td>
            <td>Inspección manual de las primeras líneas del archivo descomprimido</td>
            <td>Detección automática del separador en <code>carga_datos.py</code> antes de leer con pandas</td></tr>
        <tr><td>silhouette_score sin muestreo intentaba calcular una matriz de distancias O(n²)
            entre 300 mil o más clientes</td>
            <td>Una corrida de prueba con 300 mil filas tardó 13 minutos con uso de CPU casi nulo
            (señal de swapping de memoria, no de cómputo real)</td>
            <td>Se agregó sample_size=10.000 con semilla fija a silhouette_score; la misma corrida
            bajó a ~21 segundos</td></tr>
        <tr><td>UNIDADES constante = 1 en el 100% de los registros reales</td>
            <td>Los descriptivos mostraban desviación estándar = 0 y las pruebas devolvían NaN</td>
            <td>Se documentó como hallazgo de negocio (dataset a nivel de línea de producto) y se
            reemplazó por ITEMS POR BOLETA en la regresión y en las hipótesis relacionadas</td></tr>
        <tr><td>Edades imposibles (hasta -5.944 y 825 años) por FECHA NACIMIENTO corrupta</td>
            <td>Los estadísticos descriptivos mostraban mínimos y máximos absurdos</td>
            <td>Se definió un rango plausible [0, 100] años; fuera de rango se invalida y se imputa
            con la mediana</td></tr>
        <tr><td>Incompatibilidad de scipy.stats.kstest con la versión de scipy instalada al pasar
            args=(media, std) junto al string "norm"</td>
            <td>Excepción TypeError en la primera corrida con datos reales</td>
            <td>Se reemplazó por una distribución "frozen" (stats.norm(loc=..., scale=...).cdf)</td></tr>
        <tr><td>RAM limitada en la máquina de desarrollo (16 GB en total, ~2,5 GB libres)</td>
            <td>Monitoreo de memoria disponible antes de la corrida completa</td>
            <td>Se descartaron las columnas NOMBRES/APELLIDOS (datos personales no usados en el
            análisis) en la carga, reduciendo la huella de memoria</td></tr>
        <tr><td>Paralelizar limpieza y transformación resultó más lento que la versión secuencial
            (2,4 veces, ver sección 1), no más rápido como se esperaba inicialmente</td>
            <td>Medición directa de tiempos con y sin --parallel-preprocess sobre el dataset completo</td>
            <td>Se dejó la ruta secuencial como comportamiento por defecto; la ruta paralela se
            mantiene disponible, documentada y con particionado correcto (hash de CODIGO CLIENTE),
            para casos donde el trabajo por partición sea más pesado</td></tr>
    </table>

    <h2>7. Rigor metodológico y reproducibilidad</h2>
    <p>
    Se usó una semilla única (CPYD_SEED, valor {r['seed']}) propagada a la imputación, la partición
    train/test y la inicialización de K-Means. La elección de test estadístico en cada caso se
    condicionó a la normalidad verificada empíricamente (Shapiro-Wilk), no se asumió a priori. Todo
    hallazgo negativo (H1 y H2 no soportan la hipótesis del enunciado tal como fue planteada) se
    reporta explícitamente en vez de omitirse. La ejecución completa es reproducible con:
    <code>python analizar_ventas.py --csv data/ventas_completas.csv.gz</code> (o
    <code>--dask</code> para lectura diferida), seguido de <code>python generar_informe.py</code>
    para regenerar este informe a partir de <code>outputs/resultados.json</code>.
    </p>

    <h2>8. Conclusiones</h2>
    <p>
    El análisis sobre {fmt(n_clean,0)} transacciones válidas de Cruz Morada confirma la presencia de
    estacionalidad semanal fuerte en las ventas, diferencias significativas de ticket promedio entre
    canales y entre días de semana y fin de semana, y una relación moderada entre descuento y monto
    de compra. El modelo de regresión (R² ajustado = {fmt(reg['r_squared_adj'],3)}) y el clustering
    de clientes (silhouette = {fmt(clu['silhouette_score'],3)}) ofrecen valor descriptivo para
    segmentación y priorización comercial, aunque su poder predictivo puntual es limitado por la
    ausencia de la variable de producto. El hallazgo más relevante para la calidad de los datos
    —UNIDADES constante y ausencia total de valores nulos explícitos— muestra lo importante que es
    validar empíricamente los supuestos del enunciado contra los datos reales antes de construir el
    pipeline de análisis.
    </p>

    <div class="footer-note">
    Informe generado con <code>generar_informe.py</code> a partir de <code>outputs/resultados.json</code>
    (corrida sobre el dataset completo, semilla {r['seed']}).
    </div>

    </body></html>
    """

    with open(REPORT_PATH, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)

    if result.err:
        raise SystemExit(f"Error generando PDF ({result.err} errores)")

    print(f"Informe generado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
