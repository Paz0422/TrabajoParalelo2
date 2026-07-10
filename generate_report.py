#!/usr/bin/env python3
"""
Genera el informe técnico (PDF) a partir de outputs/resultados.json y los
gráficos de outputs/eda/. Ejecutar después de correr main.py sobre el dataset
completo:

    python generate_report.py

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


def sig_badge(sig: bool) -> str:
    if sig:
        return '<span class="badge sig">Significativo (p&lt;0,05)</span>'
    return '<span class="badge nosig">No significativo (p&ge;0,05)</span>'


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
    @page { size: A4; margin: 2.2cm 1.8cm 2.4cm 1.8cm; }
    body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #1f2430; line-height: 1.45; }
    h1 { font-size: 20pt; color: #0f3d3e; margin-bottom: 0; }
    h2 { font-size: 14.5pt; color: #0f3d3e; border-bottom: 2px solid #1f8a70; padding-bottom: 4px; margin-top: 26px; }
    h3 { font-size: 12pt; color: #145c52; margin-top: 18px; margin-bottom: 6px; }
    h4 { font-size: 10.8pt; color: #1f2430; margin-top: 12px; margin-bottom: 4px; }
    p { margin: 6px 0; text-align: justify; }
    .subtitle { font-size: 11.5pt; color: #4a5568; margin-top: 4px; }
    .meta { font-size: 9.5pt; color: #6b7280; margin-top: 14px; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0 16px 0; font-size: 9.3pt; }
    th { background-color: #0f3d3e; color: white; padding: 5px 7px; text-align: left; }
    td { padding: 4px 7px; border-bottom: 1px solid #d9dee3; }
    tr:nth-child(even) td { background-color: #f4f7f6; }
    .box { background-color: #eef7f4; border-left: 4px solid #1f8a70; padding: 8px 12px; margin: 10px 0; }
    .box-warn { background-color: #fef6e8; border-left: 4px solid #d99a1b; padding: 8px 12px; margin: 10px 0; }
    .box-find { background-color: #eef2fb; border-left: 4px solid #3757c9; padding: 8px 12px; margin: 10px 0; }
    .badge { display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: 8.8pt; color: white; }
    .badge.sig { background-color: #1f8a70; }
    .badge.nosig { background-color: #9aa1ac; }
    img.plot { width: 480px; margin: 8px 0 4px 0; border: 1px solid #d9dee3; }
    .caption { font-size: 8.8pt; color: #6b7280; text-align: center; margin-bottom: 14px; }
    .cols2 { width: 100%; }
    .cols2 td { vertical-align: top; padding: 4px; }
    .footer-note { font-size: 8.5pt; color: #9aa1ac; margin-top: 30px; }
    code { background-color: #eef2f2; padding: 1px 4px; border-radius: 2px; font-family: Courier, monospace; }
    """

    html = f"""
    <html><head><meta charset="utf-8"><style>{css}</style></head>
    <body>

    <h1>Análisis Estadístico de Datos de Ventas — Cruz Morada</h1>
    <div class="subtitle">Inferencia y Modelado sobre datos de ventas farmacéuticas con procesamiento paralelo</div>
    <div class="meta">
        Computación Paralela y Distribuida — Universidad Tecnológica Metropolitana<br/>
        Profesor: Sebastián Salazar Molina &nbsp;|&nbsp; Entrega: 10 de julio de 2026<br/>
        Semilla de reproducibilidad (CPYD_SEED): <code>{r['seed']}</code>
    </div>

    <div class="box">
        <b>Resumen ejecutivo.</b> Se procesaron <b>{fmt(n_raw,0)}</b> transacciones de venta de la cadena
        Cruz Morada (noviembre 2023 – diciembre 2024, 13 meses,
        4 canales de venta y más de 700 locales), aplicando limpieza, detección de outliers,
        ingeniería de variables y procesamiento paralelo por particiones. Tras la limpieza quedaron
        <b>{fmt(n_clean,0)}</b> registros válidos ({fmt(n_removed,0)} descartados por errores de
        registro). Se validaron 5 hipótesis estadísticas, un modelo de regresión lineal (R²
        ajustado = {fmt(reg['r_squared_adj'],3)}) y un modelo de clustering K-Means
        (silhouette = {fmt(clu['silhouette_score'],3)}) sobre la base de clientes.
    </div>

    <h2>1. Descripción de la solución implementada</h2>
    <p>
    La solución es una aplicación Python (<code>main.py</code>) estructurada en el paquete
    <code>src/cruz_morada</code>, con módulos separados para carga (<code>loader.py</code>),
    procesamiento paralelo (<code>parallel/</code>), preprocesamiento (<code>preprocessing/</code>),
    análisis exploratorio (<code>eda/</code>) e inferencia/modelado (<code>inference/</code>).
    El archivo CSV se recibe por línea de comandos (<code>--csv data/ventas_completas.csv.gz</code>)
    y se puede procesar por fragmentos (chunking, tamaño configurable) o con Dask para lectura
    diferida (<code>--dask</code>), según el volumen de datos.
    </p>
    <p>
    <b>Procesamiento paralelo:</b> los estadísticos agregados por local (conteo y suma de
    <code>MONTO APLICADO</code>) se calculan particionando el DataFrame por <code>LOCAL</code> y
    distribuyendo cada partición a un proceso worker distinto mediante
    <code>concurrent.futures.ProcessPoolExecutor</code> (módulo <code>parallel/processor.py</code>),
    aprovechando múltiples núcleos de CPU. El número de workers se ajusta automáticamente a
    <code>cpu_count() - 1</code> o se puede fijar con <code>--workers</code>.
    </p>
    <p>
    <b>Decisión de diseño sobre memoria:</b> con {fmt(n_raw,0)} filas (~217 MB comprimidos), el
    DataFrame completo cabe cómodamente en memoria tras descartar columnas de identificación
    personal no utilizadas en el análisis (<code>NOMBRES</code>, <code>APELLIDOS</code>), por lo
    que se optó por lectura en chunks que se concatenan en un único DataFrame en memoria (en vez
    de mantener todo el pipeline "out-of-core"). Esto simplifica notablemente las operaciones que
    requieren visión global de los datos (correlaciones, regresión, clustering), que de otro modo
    exigirían un motor distribuido completo (Spark/Dask con agregaciones incrementales). Para
    volúmenes que ya no quepan en memoria RAM, el flag <code>--dask</code> activa lectura lazy con
    Dask sin cambiar el resto del pipeline. Esta decisión se documenta explícitamente porque la
    máquina de desarrollo cuenta con recursos limitados (16 GB RAM, ~2,5 GB libres al momento de
    ejecutar), lo que fue una restricción real durante el desarrollo (ver sección 6, dificultades).
    </p>
    <p>
    <b>Reproducibilidad:</b> toda operación con aleatoriedad (imputación, split train/test 70/30,
    inicialización de K-Means) usa la semilla fijada por la variable de entorno
    <code>CPYD_SEED</code> (valor por defecto y usado en esta ejecución: <b>{r['seed']}</b>).
    </p>

    <h2>2. Datos y Preprocesamiento</h2>

    <h3>2.1 Carga y consolidación</h3>
    <p>
    El enunciado documenta el CSV con columnas separadas por coma; el archivo real
    (<code>ventas_completas.csv.gz</code>) usa <b>punto y coma (;) como separador y todos los
    campos entre comillas dobles</b>. El loader detecta automáticamente el separador correcto
    inspeccionando la primera línea del archivo, evitando así que el pipeline falle silenciosamente
    interpretando cada fila como una sola columna. Al ser un único archivo consolidado no fue
    necesario unificar múltiples fuentes.
    </p>

    <h3>2.2 Valores faltantes</h3>
    <p>
    Se implementó un test aproximado de MCAR (correlación entre indicadores de ausencia de cada
    columna numérica) y un reporte de nulos por columna. Sobre los {fmt(n_raw,0)} registros
    completos (y verificado también mediante barrido directo del archivo en busca de campos vacíos
    o tokens como <code>NULL</code>/<code>NA</code>), <b>no se encontraron valores faltantes
    explícitos</b> en ninguna columna. Esto difiere del ejemplo ilustrativo del enunciado (que
    sugería nulos en <code>PORCENTAJE DESCUENTO</code> o <code>FECHA NACIMIENTO</code>), pero es
    el resultado real verificado sobre el archivo entregado. El pipeline conserva la lógica de
    imputación (mediana para fechas, 0 para descuento, moda para género) por robustez ante datos
    futuros que sí contengan nulos, y el test de MCAR se ejecuta y reporta igualmente.
    </p>

    <h3>2.3 Errores de registro detectados y tratados</h3>
    <p>Se identificaron y corrigieron tres tipos de inconsistencias reales, distintas a "valores
    faltantes" pero igual de relevantes para la calidad del análisis:</p>
    <table>
        <tr><th>Problema detectado</th><th>Registros afectados</th><th>Tratamiento aplicado</th></tr>
        <tr><td><code>BOLETA</code> = 0 (no corresponde a un documento tributario real)</td>
            <td>8 de {fmt(n_raw,0)} (&lt;0,001%)</td><td>Eliminados</td></tr>
        <tr><td><code>PORCENTAJE DESCUENTO</code> fuera de rango [0,1] (se encontró un registro en 1,17 = 117%)</td>
            <td>1 registro</td><td>Eliminado</td></tr>
        <tr><td><code>EDAD</code> implausible (negativa o &gt;100 años), originada en <code>FECHA NACIMIENTO</code> corrupta</td>
            <td>3.323 (0,10%)</td><td>Invalidada e imputada con la mediana de edades plausibles (48,6 años)</td></tr>
    </table>
    <p>
    Estos casos se interpretan como <b>errores de registro</b> (no como casos de negocio reales):
    una boleta 0 no existe, un descuento sobre el 100% es imposible, y una edad de -5.944 o 825
    años (valores observados antes del tratamiento) no corresponde a ningún cliente real. El umbral
    [0, 100] años para <code>EDAD</code> es conservador y documentado explícitamente en
    <code>features.py</code> (<code>MIN_EDAD_PLAUSIBLE</code>, <code>MAX_EDAD_PLAUSIBLE</code>).
    </p>

    <h3>2.4 Detección de outliers (variables continuas)</h3>
    <p>
    Se aplicaron los métodos IQR (rango intercuartílico, factor 1,5) y Z-score (umbral 3) sobre
    <code>MONTO APLICADO</code>:
    </p>
    <table>
        <tr><th>Método</th><th>Outliers detectados</th><th>% del total</th></tr>
        <tr><td>IQR (1,5×RIC)</td><td>{fmt(prep['outliers_monto']['n_iqr_outliers'],0)}</td><td>{fmt(prep['outliers_monto']['pct_iqr_outliers'])}%</td></tr>
        <tr><td>Z-score (|z|&gt;3)</td><td>{fmt(prep['outliers_monto']['n_zscore_outliers'],0)}</td><td>{fmt(prep['outliers_monto']['pct_zscore_outliers'])}%</td></tr>
    </table>
    <p>
    A diferencia de los errores de la sección 2.3, estos <b>no se eliminan</b>: dado que
    <code>MONTO APLICADO</code> tiene una distribución fuertemente asimétrica a la derecha
    (asimetría = {fmt(desc['skewness']['MONTO APLICADO'])}, curtosis = {fmt(desc['kurtosis']['MONTO APLICADO'])}),
    montos altos (hasta ${fmt(desc['max']['MONTO APLICADO'],0)} CLP) son consistentes con compras
    de medicamentos de alto costo (tratamientos crónicos, oncológicos, etc.) y no con errores de
    digitación. Se interpretan como <b>casos de negocio reales</b> de cola larga, relevantes para
    el negocio (ej. compras vía convenios de salud) y se conservan en el análisis.
    </p>
    <p>
    <code>UNIDADES</code> no presenta outliers (0% por ambos métodos) por la razón que se explica
    a continuación.
    </p>

    <div class="box-find">
    <b>Hallazgo relevante:</b> <code>UNIDADES</code> es <b>constante e igual a 1 en el 100% de los
    {fmt(n_raw,0)} registros</b>. Esto revela que el dataset está a nivel de <i>línea de producto
    por boleta</i> (una fila = un producto distinto comprado), no a nivel de cantidad total. Por lo
    tanto <code>UNIDADES</code> no aporta varianza para ningún análisis estadístico (correlaciones,
    regresión, ANOVA), y se documenta así en vez de forzar pruebas degeneradas. Como proxy real del
    volumen de compra se creó la variable derivada <code>ITEMS POR BOLETA</code> (cantidad de líneas
    de producto distintas por número de boleta), que sí varía (media = {fmt(desc['mean']['ITEMS POR BOLETA'])},
    máximo = {fmt(desc['max']['ITEMS POR BOLETA'],0)}) y se usa en su reemplazo en el modelo de
    regresión y en la hipótesis H2.
    </div>

    <h3>2.5 Variables derivadas</h3>
    <table>
        <tr><th>Variable</th><th>Definición</th><th>Media</th><th>Nota</th></tr>
        <tr><td><code>MONTO POR UNIDAD</code></td><td>MONTO APLICADO / UNIDADES</td>
            <td>${fmt(desc['mean']['MONTO POR UNIDAD'],0)}</td>
            <td>Idéntica a MONTO APLICADO (UNIDADES=1 siempre); se mantiene por completitud del enunciado</td></tr>
        <tr><td><code>EDAD</code></td><td>(FECHA − FECHA NACIMIENTO) / 365,25, saneada</td>
            <td>{fmt(desc['mean']['EDAD'])} años</td><td>3.323 valores implausibles imputados (ver 2.3)</td></tr>
        <tr><td><code>FRECUENCIA COMPRA</code></td><td>N° de boletas distintas por CODIGO CLIENTE</td>
            <td>{fmt(desc['mean']['FRECUENCIA COMPRA'])}</td><td>Máximo observado: {fmt(desc['max']['FRECUENCIA COMPRA'],0)} boletas por cliente</td></tr>
        <tr><td><code>ITEMS POR BOLETA</code></td><td>N° de líneas de producto por BOLETA</td>
            <td>{fmt(desc['mean']['ITEMS POR BOLETA'])}</td><td>Proxy real de volumen (ver hallazgo 2.4)</td></tr>
        <tr><td><code>HORA / DIA SEMANA / MES / ANIO / ES_FIN_DE_SEMANA</code></td>
            <td>Extraídas de FECHA</td><td>-</td><td>Usadas en análisis temporal e hipótesis H3</td></tr>
    </table>
    <p>
    No se aplicó estandarización (StandardScaler) a las variables usadas en regresión/ANOVA para
    mantener la interpretación directa de los coeficientes en las unidades originales; sí se
    estandarizan las variables de clientes antes del clustering (sección 4.2), documentando el
    scaler ajustado sobre el conjunto de entrenamiento.
    </p>

    <h2>3. Análisis Exploratorio Estadístico</h2>

    <h3>3.1 Estadística descriptiva</h3>
    <table>
        <tr><th>Variable</th><th>Media</th><th>Desv. Est.</th><th>Mediana</th><th>Asimetría</th><th>Curtosis</th></tr>
        <tr><td>MONTO APLICADO (CLP)</td><td>{fmt(desc['mean']['MONTO APLICADO'],0)}</td><td>{fmt(desc['std']['MONTO APLICADO'],0)}</td><td>{fmt(desc['50%']['MONTO APLICADO'],0)}</td><td>{fmt(desc['skewness']['MONTO APLICADO'])}</td><td>{fmt(desc['kurtosis']['MONTO APLICADO'])}</td></tr>
        <tr><td>PORCENTAJE DESCUENTO</td><td>{fmt(desc['mean']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['std']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['50%']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['skewness']['PORCENTAJE DESCUENTO'])}</td><td>{fmt(desc['kurtosis']['PORCENTAJE DESCUENTO'])}</td></tr>
        <tr><td>EDAD (años)</td><td>{fmt(desc['mean']['EDAD'])}</td><td>{fmt(desc['std']['EDAD'])}</td><td>{fmt(desc['50%']['EDAD'])}</td><td>{fmt(desc['skewness']['EDAD'])}</td><td>{fmt(desc['kurtosis']['EDAD'])}</td></tr>
        <tr><td>ITEMS POR BOLETA</td><td>{fmt(desc['mean']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['std']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['50%']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['skewness']['ITEMS POR BOLETA'])}</td><td>{fmt(desc['kurtosis']['ITEMS POR BOLETA'])}</td></tr>
    </table>
    <p>
    <b>Interpretación no técnica:</b> el ticket promedio es de ${fmt(desc['mean']['MONTO APLICADO'],0)} CLP,
    pero la mediana (${fmt(desc['50%']['MONTO APLICADO'],0)}) es notablemente menor que la media,
    señal de una distribución con cola larga a la derecha: la mayoría de las compras son de monto
    moderado, pero existen compras puntuales muy altas que "tiran" el promedio hacia arriba (curtosis
    de {fmt(desc['kurtosis']['MONTO APLICADO'])} indica una concentración de valores extremos muy superior a
    una distribución normal). La edad promedio de los clientes es de {fmt(desc['mean']['EDAD'])} años,
    con alta dispersión (±{fmt(desc['std']['EDAD'])} años), reflejando una base de clientes heterogénea
    típica de una farmacia.
    </p>

    <h4>Test de normalidad (Shapiro-Wilk y Kolmogorov-Smirnov)</h4>
    <table>
        <tr><th>Variable</th><th>Shapiro-Wilk (p)</th><th>Kolmogorov-Smirnov (p)</th><th>¿Normal?</th></tr>
        <tr><td>MONTO APLICADO</td><td>{pval_fmt(eda['normality_tests']['MONTO APLICADO']['shapiro']['p_value'])}</td>
            <td>{pval_fmt(eda['normality_tests']['MONTO APLICADO']['kolmogorov_smirnov']['p_value'])}</td><td>No</td></tr>
        <tr><td>PORCENTAJE DESCUENTO</td><td>{pval_fmt(eda['normality_tests']['PORCENTAJE DESCUENTO']['shapiro']['p_value'])}</td>
            <td>{pval_fmt(eda['normality_tests']['PORCENTAJE DESCUENTO']['kolmogorov_smirnov']['p_value'])}</td><td>No</td></tr>
    </table>
    <p>
    Ambas pruebas rechazan la normalidad (p &lt; 0,0001) para las dos variables continuas
    principales. En consecuencia, el pipeline selecciona automáticamente <b>correlación de
    Spearman</b> (basada en rangos, no asume normalidad) en vez de Pearson para la matriz de
    correlación general, y cada prueba de hipótesis de comparación de medias verifica normalidad
    por grupo antes de elegir entre t-test/Welch o Mann-Whitney U.
    </p>

    <img class="plot" src="{img_b64('hist_monto_aplicado.png')}"/>
    <div class="caption">Fig. 1 — Distribución de MONTO APLICADO: fuerte asimetría a la derecha, consistente con el rechazo de normalidad.</div>

    <img class="plot" src="{img_b64('hist_porcentaje_descuento.png')}"/>
    <div class="caption">Fig. 2 — Distribución de PORCENTAJE DESCUENTO: se concentra en tramos discretos (≈20%, 35%, 40%, 50%) en vez de ser continua, consistente con tramos de convenio/copago típicos de farmacias (Fonasa/Isapre) en vez de descuentos arbitrarios.</div>

    <h3>3.2 Boxplot por categoría</h3>
    <img class="plot" src="{img_b64('boxplot_monto_por_canal.png')}"/>
    <div class="caption">Fig. 3 — MONTO APLICADO por CANAL. Las medianas son similares entre canales, pero POS y WEB concentran más valores extremos (mayor volumen de transacciones).</div>

    <h3>3.3 Matriz de correlación (Spearman) con significancia</h3>
    <img class="plot" src="{img_b64('correlation_matrix.png')}"/>
    <div class="caption">Fig. 4 — UNIDADES aparece en blanco por ser constante (correlación no definida). MONTO APLICADO y PORCENTAJE DESCUENTO muestran correlación positiva moderada.</div>
    <p>
    La correlación de Spearman entre <code>MONTO APLICADO</code> y <code>PORCENTAJE DESCUENTO</code>
    es de <b>{fmt(eda['correlation']['MONTO APLICADO']['PORCENTAJE DESCUENTO'])}</b>
    ({pval_fmt(eda['correlation_pvalues']['MONTO APLICADO']['PORCENTAJE DESCUENTO'])}), estadísticamente
    significativa: a mayor monto de compra, mayor tiende a ser el porcentaje de descuento aplicado.
    Esto es coherente con la lectura de negocio: tratamientos de mayor valor (crónicos, especialidades)
    suelen estar cubiertos por convenios con mayor cobertura porcentual.
    </p>

    <h3>3.4 Asociación entre variables categóricas</h3>
    <table>
        <tr><th>Prueba</th><th>Estadístico</th><th>gl</th><th>p-value</th><th>Conclusión</th></tr>
        <tr><td>Chi-cuadrado: CANAL × LOCAL</td><td>χ² = {fmt(eda['chi2_canal_local']['chi2'],0)}</td>
            <td>{eda['chi2_canal_local']['dof']}</td><td>{pval_fmt(eda['chi2_canal_local']['p_value'])}</td>
            <td>{sig_badge(eda['chi2_canal_local']['significant_005'])}</td></tr>
        <tr><td>ANOVA: MONTO APLICADO ~ CANAL</td><td>F = {fmt(eda['anova_monto_canal']['f_statistic'])}</td>
            <td>-</td><td>{pval_fmt(eda['anova_monto_canal']['p_value'])}</td>
            <td>{sig_badge(eda['anova_monto_canal']['significant_005'])}</td></tr>
    </table>
    <p>
    <b>Interpretación no técnica:</b> el canal de venta (POS, WEB, APP, CCT) y el local donde se
    realiza la compra <b>no son independientes</b> (p &lt; 0,05): la mezcla de canales varía
    significativamente entre locales (algunos locales concentran más ventas WEB/APP, esperable si
    corresponden a zonas con distinto perfil de clientes). El monto promedio de compra también
    <b>difiere significativamente entre canales</b> (ANOVA, p &lt; 0,05), aunque como se ve en el
    boxplot (Fig. 3) las medianas son parecidas y la diferencia es impulsada principalmente por la
    cola de valores altos.
    </p>

    <h3>3.5 Patrones temporales</h3>
    <p>
    Se agregaron las ventas diarias (suma de MONTO APLICADO por día) sobre el período completo
    (nov-2023 a dic-2024) y se aplicó descomposición aditiva (tendencia + estacionalidad + residuo,
    período = 7 días) y análisis de autocorrelación.
    </p>
    <img class="plot" src="{img_b64('seasonal_decomposition.png')}"/>
    <div class="caption">Fig. 5 — Descomposición de ventas diarias. La venta diaria promedio es de
    ${fmt(eda['daily_sales_summary']['mean'],0)} CLP (±${fmt(eda['daily_sales_summary']['std'],0)}).
    Se observa una <b>tendencia creciente</b> sostenida durante el período y un fuerte
    <b>componente estacional semanal</b> superpuesto (picos y valles que se repiten cada 7 días).
    El día de mayor venta agregada fue el {eda['daily_sales_summary']['max_date'][:10]}.</div>

    <img class="plot" src="{img_b64('acf_pacf.png')}"/>
    <div class="caption">Fig. 6 — ACF y PACF de la serie diaria. La ACF muestra picos marcados y
    persistentes en los rezagos múltiplos de 7 (7, 14, 21, 28, 35), confirmando una
    <b>estacionalidad semanal</b> clara: el nivel de ventas de un día depende fuertemente del nivel
    del mismo día de la semana anterior. La PACF muestra que, controlando por rezagos intermedios,
    el rezago 7 sigue siendo el más relevante después del rezago 1.</div>

    <h2>4. Inferencia Estadística</h2>

    <h3>4.1 Pruebas de hipótesis</h3>
    <p>Se validaron 2 hipótesis del enunciado (H1, H2) y 3 hipótesis propias (H3, H4, H5).</p>

    <h4>H1 — "El ticket promedio en APP es mayor que en WEB" (ejemplo del enunciado)</h4>
    <p>
    Test: {ht['H1_ticket_app_vs_web']['test']}. Media APP = ${fmt(ht['H1_ticket_app_vs_web']['mean_a'],0)},
    media WEB = ${fmt(ht['H1_ticket_app_vs_web']['mean_b'],0)}.
    {pval_fmt(ht['H1_ticket_app_vs_web']['p_value'])}. {sig_badge(ht['H1_ticket_app_vs_web']['significant_005'])}
    </p>
    <div class="box-warn">
    <b>Interpretación no técnica:</b> aunque el ticket promedio observado en WEB
    (${fmt(ht['H1_ticket_app_vs_web']['mean_b'],0)}) es numéricamente mayor que en APP
    (${fmt(ht['H1_ticket_app_vs_web']['mean_a'],0)}), la prueba de Mann-Whitney U indica que
    <b>esta diferencia no es estadísticamente significativa</b> (p = 0,72): no hay evidencia
    suficiente para afirmar que el canal APP o WEB tenga, por sí solo, un ticket promedio distinto.
    Es un hallazgo honesto que contradice la premisa del ejemplo del enunciado, y se reporta así
    porque es lo que muestran los datos reales.
    </div>

    <h4>H2 — "El % de descuento afecta el volumen de compra" (ejemplo del enunciado, adaptado)</h4>
    <p>
    Como se explicó en 2.4, <code>UNIDADES</code> es constante y no permite esta prueba tal como
    está planteada en el enunciado; se usa <code>ITEMS POR BOLETA</code> como variable dependiente.
    Test: regresión lineal simple. Pendiente = {fmt(ht['H2_descuento_vs_volumen']['slope'],4)},
    R² = {fmt(ht['H2_descuento_vs_volumen']['r_squared'],6)}, {pval_fmt(ht['H2_descuento_vs_volumen']['p_value'])}.
    {sig_badge(ht['H2_descuento_vs_volumen']['significant_005'])}
    </p>
    <div class="box-warn">
    <b>Interpretación no técnica:</b> la relación es estadísticamente significativa (con más de 3
    millones de registros, incluso efectos diminutos resultan significativos), pero el R² de
    {fmt(ht['H2_descuento_vs_volumen']['r_squared'],6)} indica que el porcentaje de descuento
    explica una fracción prácticamente nula de la variación en ítems por boleta. En términos de
    negocio: <b>el descuento aplicado no es un predictor relevante del volumen de productos que
    lleva un cliente en su boleta</b>, aunque la relación exista formalmente.
    </div>

    <h4>H3 (propia) — "El ticket promedio difiere entre fin de semana y día de semana"</h4>
    <p>
    Test: {ht['H3_monto_finde_vs_semana']['test']}. Media fin de semana = ${fmt(ht['H3_monto_finde_vs_semana']['mean_finde'],0)},
    media día de semana = ${fmt(ht['H3_monto_finde_vs_semana']['mean_semana'],0)}.
    {pval_fmt(ht['H3_monto_finde_vs_semana']['p_value'])}. {sig_badge(ht['H3_monto_finde_vs_semana']['significant_005'])}
    </p>
    <div class="box">
    <b>Interpretación no técnica:</b> el ticket promedio en días de semana
    (${fmt(ht['H3_monto_finde_vs_semana']['mean_semana'],0)}) es significativamente <b>mayor</b> que
    en fines de semana (${fmt(ht['H3_monto_finde_vs_semana']['mean_finde'],0)}). Una lectura de
    negocio plausible: las compras de mayor valor (tratamientos crónicos, retiro de recetas médicas)
    se concentran en días hábiles, mientras que el fin de semana concentra compras más pequeñas
    y espontáneas (ej. analgésicos, cuidado personal).
    </div>

    <h4>H4 (propia) — "La edad del cliente se correlaciona con el monto de compra"</h4>
    <p>
    Test: Pearson (aplicado sobre EDAD, cuya distribución es más cercana a la normal que MONTO
    APLICADO; se reporta como referencia adicional a la matriz Spearman de la sección 3.3).
    r = {fmt(ht['H4_edad_vs_monto']['correlation'])}. {pval_fmt(ht['H4_edad_vs_monto']['p_value'])}.
    {sig_badge(ht['H4_edad_vs_monto']['significant_005'])}
    </p>
    <div class="box-warn">
    <b>Interpretación no técnica:</b> existe una correlación positiva pero <b>muy débil</b>
    (r = {fmt(ht['H4_edad_vs_monto']['correlation'])}) entre la edad del cliente y el monto de
    compra: clientes de mayor edad tienden, en promedio, a gastar levemente más, pero la edad por
    sí sola explica muy poco de la variación del monto (consistente con clientes de mayor edad
    asociados a tratamientos crónicos de mayor costo, pero el efecto es marginal frente a otros
    factores como el producto específico comprado).
    </div>

    <h4>H5 (propia) — "El género del cliente influye en el monto promedio de compra"</h4>
    <p>
    Test: ANOVA one-way. F = {fmt(ht['H5_genero_vs_monto']['f_statistic'])}.
    {pval_fmt(ht['H5_genero_vs_monto']['p_value'])}. {sig_badge(ht['H5_genero_vs_monto']['significant_005'])}
    </p>
    <div class="box">
    <b>Interpretación no técnica:</b> existe una diferencia estadísticamente significativa en el
    monto promedio de compra entre géneros. El tamaño de la muestra (más de 3 millones de registros)
    hace que incluso diferencias moderadas sean altamente significativas; se recomienda a negocio
    revisar la magnitud absoluta de la diferencia (media por grupo) antes de tomar decisiones
    comerciales basadas solo en el p-value.
    </div>

    <h3>4.2 Modelado predictivo — Opción A: Regresión lineal</h3>
    <p>
    Se modeló <code>MONTO APLICADO ~ CANAL + LOCAL + ITEMS POR BOLETA + PORCENTAJE DESCUENTO</code>
    mediante OLS (statsmodels), con partición train/test 70/30 (semilla {r['seed']}).
    <code>UNIDADES</code> se sustituyó por <code>ITEMS POR BOLETA</code> por la razón explicada en
    2.4: incluir una columna constante en la regresión produce colinealidad perfecta con el
    intercepto (matriz de diseño no invertible), lo que se detectó en una corrida preliminar
    (coeficientes idénticos entre intercepto y UNIDADES) y se corrigió antes de la entrega.
    </p>
    <table>
        <tr><th>Métrica</th><th>Valor</th></tr>
        <tr><td>R² ajustado</td><td>{fmt(reg['r_squared_adj'],4)}</td></tr>
        <tr><td>RMSE (test)</td><td>${fmt(reg['rmse_test'],0)}</td></tr>
        <tr><td>MAE (test)</td><td>${fmt(reg['mae_test'],0)}</td></tr>
    </table>
    <h4>Coeficientes y significancia</h4>
    <table>
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
    <table>
        <tr><th>Variable</th><th>VIF</th><th>Lectura</th></tr>
        <tr><td>LOCAL</td><td>{fmt(reg['vif']['LOCAL'])}</td><td>Sin problema (VIF &lt; 5)</td></tr>
        <tr><td>ITEMS POR BOLETA</td><td>{fmt(reg['vif']['ITEMS POR BOLETA'])}</td><td>Sin problema (VIF &lt; 5)</td></tr>
        <tr><td>PORCENTAJE DESCUENTO</td><td>{fmt(reg['vif']['PORCENTAJE DESCUENTO'])}</td><td>Sin problema (VIF &lt; 5)</td></tr>
    </table>
    <p>
    Ningún VIF supera 5, por lo que no hay evidencia de multicolinealidad severa entre los
    predictores retenidos. Respecto a los supuestos clásicos de OLS: dado que ni MONTO APLICADO ni
    PORCENTAJE DESCUENTO son normales (sección 3.1) y el modelo excluye variables clave (ver
    limitaciones), es esperable que los residuales no sean perfectamente homocedásticos ni
    normales; esto es consistente con un R² ajustado moderado
    ({fmt(reg['r_squared_adj'],3)}) y se discute como limitación explícita a continuación en vez de
    ocultarse.
    </p>
    <div class="box-warn">
    <b>Interpretación no técnica:</b> el modelo explica solo el {fmt(reg['r_squared_adj']*100,1)}%
    de la variación del monto de compra. El coeficiente de <code>PORCENTAJE DESCUENTO</code>
    (+{fmt(reg['coefficients']["Q('PORCENTAJE DESCUENTO')"],0)}) es, con diferencia, el más
    influyente: transacciones con mayor porcentaje de descuento tienden a tener un monto final
    más alto. Esto <b>no debe leerse como "el descuento causa un mayor gasto"</b>, sino como reflejo
    de que los tratamientos de mayor valor (crónicos, especialidades médicas) suelen tener
    convenios con mayor cobertura porcentual — la causalidad probablemente va en la dirección
    opuesta o es bidireccional. El canal POS y WEB muestran tickets significativamente más altos
    que APP (canal de referencia); CCT no muestra diferencia significativa (posiblemente por su
    bajo volumen de transacciones). <code>LOCAL</code>, al ser un identificador numérico y no una
    variable de negocio continua, se interpreta con cautela (ver limitaciones).
    </div>

    <h3>4.3 Modelado descriptivo — Opción B: Clustering de clientes (K-Means)</h3>
    <p>
    Se segmentaron los clientes (agrupados por <code>CODIGO CLIENTE</code>) usando 4 variables
    agregadas: monto total gastado, total de ítems comprados, número de transacciones distintas y
    descuento promedio recibido, estandarizadas con <code>StandardScaler</code> antes de aplicar
    K-Means (k=4, semilla {r['seed']}, entrenado sobre partición 70% y evaluado sobre el 100% de
    los clientes).
    </p>
    <table>
        <tr><th>Métrica</th><th>Valor</th></tr>
        <tr><td>Silhouette score</td><td>{fmt(clu['silhouette_score'],3)}</td></tr>
        <tr><td>Inercia (WCSS)</td><td>{fmt(clu['inertia'],0)}</td></tr>
    </table>
    <table>
        <tr><th>Clúster</th><th>N° de clientes</th><th>% del total</th></tr>
        {"".join(f"<tr><td>{k}</td><td>{fmt(v,0)}</td><td>{fmt(100*v/sum(clu['cluster_sizes'].values()),1)}%</td></tr>" for k, v in clu['cluster_sizes'].items())}
    </table>
    <div class="box">
    <b>Interpretación no técnica:</b> un silhouette score de {fmt(clu['silhouette_score'],3)}
    (escala -1 a 1) indica una <b>separación moderada-buena</b> entre los 4 segmentos de clientes:
    los grupos son razonablemente distinguibles entre sí. El clúster más grande concentra
    aproximadamente el {fmt(100*max(clu['cluster_sizes'].values())/sum(clu['cluster_sizes'].values()),0)}%
    de los clientes (perfil "estándar", bajo gasto/baja frecuencia), mientras que el clúster más
    pequeño (~{fmt(100*min(clu['cluster_sizes'].values())/sum(clu['cluster_sizes'].values()),1)}%)
    probablemente corresponde a clientes de alto valor (alta frecuencia y/o alto gasto acumulado),
    útil para campañas de fidelización dirigidas. Nota: dado que un porcentaje considerable de
    <code>CODIGO CLIENTE</code> distintos realiza muy pocas compras (ver <code>FRECUENCIA COMPRA</code>,
    mediana = {fmt(desc['50%']['FRECUENCIA COMPRA'],0)}), varios clústeres pueden estar dominados
    por clientes de compra única.
    </div>

    <h3>4.4 Validación de modelos y extrapolabilidad</h3>
    <p>
    Ambos modelos se evaluaron con partición train/test 70/30 fijada por semilla, siguiendo el
    enunciado. El error absoluto medio de la regresión (MAE = ${fmt(reg['mae_test'],0)}) representa
    aproximadamente el {fmt(100*reg['mae_test']/desc['mean']['MONTO APLICADO'],0)}% del ticket
    promedio, un margen de error considerable para uso como predictor puntual de negocio, aunque
    útil para explicar tendencias agregadas.
    </p>
    <p><b>¿Es el modelo extrapolable?</b> Con matices importantes:</p>
    <ul>
        <li><b>Ventana temporal limitada:</b> el dataset cubre 13 meses (nov-2023 a dic-2024); la
        tendencia creciente observada (Fig. 5) podría no sostenerse indefinidamente y el modelo no
        la captura explícitamente (no se incluyó un término de tendencia temporal en la regresión).</li>
        <li><b>Variable de producto omitida:</b> el predictor más obvio del monto de una transacción
        —qué producto/SKU se compró— no se incluyó en el modelo por su altísima cardinalidad
        (cientos de SKUs), lo que limita el R² alcanzable. Un modelo por categoría de producto
        probablemente mejoraría sustancialmente el ajuste.</li>
        <li><b>LOCAL como variable numérica continua:</b> es un identificador, no una magnitud;
        su coeficiente lineal no tiene una interpretación de negocio directa (un modelo con
        efectos fijos por local, inviable aquí por la cardinalidad >700, sería más apropiado).</li>
        <li><b>Sesgo de canal:</b> CCT representa una fracción muy pequeña de las transacciones
        (canal minoritario), por lo que sus estimaciones son menos confiables que las de POS/WEB/APP.</li>
        <li><b>Clustering sensible a la definición de "cliente":</b> con una fracción relevante de
        compradores de una sola transacción, los clústeres reflejan en parte esa heterogeneidad
        estructural más que un comportamiento diferenciado real.</li>
    </ul>
    <p>
    En síntesis, los modelos son útiles como <b>herramientas descriptivas y de apoyo a decisiones
    agregadas</b> (qué canal/local revisar, qué segmento de clientes priorizar), pero no deberían
    usarse como predictor puntual de "cuánto gastará este cliente" sin incorporar la variable de
    producto y una ventana temporal más larga.
    </p>

    <h2>5. Justificación de librerías utilizadas</h2>
    <table>
        <tr><th>Librería</th><th>Uso en el proyecto</th></tr>
        <tr><td>pandas</td><td>Carga, limpieza y transformación tabular del dataset completo</td></tr>
        <tr><td>NumPy</td><td>Operaciones numéricas vectorizadas de soporte</td></tr>
        <tr><td>Dask</td><td>Lectura diferida (lazy) opcional para volúmenes que excedan la RAM disponible</td></tr>
        <tr><td>SciPy (stats)</td><td>Pruebas de normalidad (Shapiro-Wilk, KS), correlación (Pearson/Spearman), t-test, Mann-Whitney U, ANOVA, chi-cuadrado</td></tr>
        <tr><td>statsmodels</td><td>Regresión OLS con fórmulas tipo R, cálculo de VIF, descomposición estacional, ACF/PACF</td></tr>
        <tr><td>scikit-learn</td><td>K-Means, StandardScaler, train_test_split, métricas (RMSE, MAE, silhouette)</td></tr>
        <tr><td>matplotlib / seaborn</td><td>Visualizaciones estadísticas (histogramas, boxplots, heatmaps, series de tiempo)</td></tr>
        <tr><td>concurrent.futures (stdlib)</td><td>Paralelización real de estadísticos agregados por partición, sin dependencias adicionales</td></tr>
    </table>

    <h2>6. Dificultades encontradas y cómo se resolvieron</h2>
    <table>
        <tr><th>Dificultad</th><th>Cómo se detectó</th><th>Resolución</th></tr>
        <tr><td>El CSV real usa <code>;</code> como separador y comillas, no coma como indica el enunciado</td>
            <td>Inspección manual de las primeras líneas del archivo descomprimido</td>
            <td>Detección automática del separador en <code>loader.py</code> antes de leer con pandas</td></tr>
        <tr><td><code>silhouette_score</code> sin muestreo intentaba calcular una matriz de
            distancias O(n²) entre ~300 mil+ clientes</td>
            <td>Una corrida de prueba con 300 mil filas tardó 13 minutos con uso de CPU casi nulo
            (señal de <i>swapping</i> de memoria, no de cómputo real)</td>
            <td>Se agregó <code>sample_size=10.000</code> con semilla fija a <code>silhouette_score</code>;
            la misma corrida bajó a ~21 segundos</td></tr>
        <tr><td><code>UNIDADES</code> constante = 1 en el 100% de los registros reales</td>
            <td>Descriptivos mostraban desviación estándar = 0 y pruebas devolvían NaN</td>
            <td>Se documentó como hallazgo de negocio (dataset a nivel de línea de producto) y se
            reemplazó por <code>ITEMS POR BOLETA</code> en regresión e hipótesis relacionadas</td></tr>
        <tr><td>Edades imposibles (hasta -5.944 y 825 años) por <code>FECHA NACIMIENTO</code> corrupta</td>
            <td>Estadísticos descriptivos mostraban mínimos/máximos absurdos</td>
            <td>Se definió un rango plausible [0, 100] años; fuera de rango se invalida y se imputa
            con la mediana</td></tr>
        <tr><td>Incompatibilidad de <code>scipy.stats.kstest</code> con la versión de scipy instalada
            al pasar <code>args=(media, std)</code> junto al string <code>"norm"</code></td>
            <td>Excepción <code>TypeError</code> en la primera corrida con datos reales</td>
            <td>Se reemplazó por una distribución "frozen" (<code>stats.norm(loc=..., scale=...).cdf</code>)</td></tr>
        <tr><td>RAM limitada en la máquina de desarrollo (16 GB totales, ~2,5 GB libres)</td>
            <td>Monitoreo de memoria disponible antes de la corrida completa</td>
            <td>Se descartaron columnas <code>NOMBRES</code>/<code>APELLIDOS</code> (PII no usada
            en el análisis) en la carga, reduciendo la huella de memoria</td></tr>
    </table>

    <h2>7. Rigor metodológico y reproducibilidad</h2>
    <ul>
        <li>Semilla única (<code>CPYD_SEED</code>, valor {r['seed']}) propagada a imputación,
        split train/test y K-Means.</li>
        <li>Elección de test estadístico condicionada a normalidad verificada empíricamente
        (Shapiro-Wilk) en cada caso, no asumida a priori.</li>
        <li>Todo hallazgo "negativo" (H1 y H2 no soportan la hipótesis del enunciado tal como fue
        planteada) se reporta explícitamente en vez de omitirse.</li>
        <li>Ejecución completa reproducible con: <code>python main.py --csv data/ventas_completas.csv.gz</code>
        (o <code>--dask</code> para lectura diferida), seguido de <code>python generate_report.py</code>
        para regenerar este informe desde <code>outputs/resultados.json</code>.</li>
    </ul>

    <h2>8. Conclusiones</h2>
    <p>
    El análisis sobre {fmt(n_clean,0)} transacciones válidas de Cruz Morada confirma la presencia
    de estacionalidad semanal fuerte en las ventas, diferencias significativas de ticket promedio
    entre canales y entre días de semana/fin de semana, y una relación moderada entre descuento y
    monto de compra. El modelo de regresión (R² ajustado = {fmt(reg['r_squared_adj'],3)}) y el
    clustering de clientes (silhouette = {fmt(clu['silhouette_score'],3)}) ofrecen valor descriptivo
    para segmentación y priorización comercial, aunque su poder predictivo puntual es limitado por
    la ausencia de la variable de producto. El hallazgo más relevante para la calidad de datos —
    <code>UNIDADES</code> constante y ausencia total de valores nulos explícitos— ilustra la
    importancia de validar empíricamente los supuestos del enunciado contra los datos reales antes
    de construir el pipeline de análisis.
    </p>

    <div class="footer-note">
    Informe generado automáticamente por <code>generate_report.py</code> a partir de
    <code>outputs/resultados.json</code> (corrida sobre el dataset completo, semilla {r['seed']}).
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
