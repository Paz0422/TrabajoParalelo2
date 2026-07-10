"""Modelado predictivo y descriptivo."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from cruz_morada.configuracion import OUTPUT_DIR, SEED, TRAIN_TEST_RATIO

logger = logging.getLogger(__name__)


def run_regression_model(df: pd.DataFrame) -> dict:
    """
    Opción A: Regresión de MONTO APLICADO ~ CANAL + LOCAL + volumen + PORCENTAJE DESCUENTO.
    Incluye diagnóstico VIF y métricas train/test 70/30.

    UNIDADES es constante (=1) en el 100% de los registros reales: incluirla como
    predictor produce colinealidad perfecta con el intercepto (matriz de diseño no
    invertible). Se usa ITEMS POR BOLETA (líneas de producto por transacción) como
    proxy real de volumen de compra.
    """
    volume_col = "ITEMS POR BOLETA" if "ITEMS POR BOLETA" in df.columns else "UNIDADES"
    formula = f"Q('MONTO APLICADO') ~ C(CANAL) + LOCAL + Q('{volume_col}') + Q('PORCENTAJE DESCUENTO')"
    sub = df.dropna(subset=["MONTO APLICADO", "CANAL", "LOCAL", volume_col, "PORCENTAJE DESCUENTO"])

    train, test = train_test_split(sub, train_size=TRAIN_TEST_RATIO, random_state=SEED)

    model = smf.ols(formula, data=train).fit()
    predictions = model.predict(test)

    rmse = float(np.sqrt(mean_squared_error(test["MONTO APLICADO"], predictions)))
    mae = float(mean_absolute_error(test["MONTO APLICADO"], predictions))

    # VIF sobre variables numéricas
    vif_data = _compute_vif(train, volume_col)

    # Diagnóstico de supuestos: homocedasticidad (Breusch-Pagan) y normalidad de
    # residuales (Jarque-Bera), más gráficos de residuos vs. ajustados y Q-Q.
    diagnostics = _diagnose_assumptions(model)

    return {
        "model_type": "OLS Regression",
        "r_squared_adj": float(model.rsquared_adj),
        "rmse_test": rmse,
        "mae_test": mae,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "vif": vif_data,
        "diagnostics": diagnostics,
        "summary_excerpt": str(model.summary().tables[1]),
    }


def _diagnose_assumptions(model, output_dir: Path | None = None) -> dict:
    """
    Diagnóstico de supuestos de OLS sobre el conjunto de entrenamiento:
    - Homocedasticidad: test de Breusch-Pagan (H0: varianza constante de residuales).
    - Normalidad de residuales: test de Jarque-Bera (H0: residuales distribuidos normal).
    - Linealidad: inspección visual del gráfico de residuos vs. valores ajustados
      (un patrón sistemático, no una nube sin forma, indicaría mala especificación).
    """
    resid = model.resid
    fitted = model.fittedvalues

    bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, model.model.exog)
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid)

    out = output_dir or OUTPUT_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    # Muestra para los gráficos: graficar 2M+ puntos es lento y poco legible.
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(resid), size=min(5000, len(resid)), replace=False)
    resid_sample = resid.iloc[sample_idx]
    fitted_sample = fitted.iloc[sample_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(fitted_sample, resid_sample, alpha=0.3, s=10)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Valores ajustados")
    ax.set_ylabel("Residuales")
    ax.set_title("Residuos vs. valores ajustados (muestra de 5.000 puntos)")
    plt.tight_layout()
    plt.savefig(out / "regresion_residuos_vs_ajustados.png", dpi=150)
    plt.close()

    fig = qqplot(resid_sample, line="s")
    fig.set_size_inches(8, 5)
    plt.title("Q-Q plot de residuales (muestra de 5.000 puntos)")
    plt.tight_layout()
    plt.savefig(out / "regresion_qqplot_residuales.png", dpi=150)
    plt.close()

    return {
        "breusch_pagan": {
            "statistic": float(bp_stat),
            "p_value": float(bp_pvalue),
            "homocedastico": bool(bp_pvalue > 0.05),
        },
        "jarque_bera": {
            "statistic": float(jb_stat),
            "p_value": float(jb_pvalue),
            "skew": float(skew),
            "kurtosis": float(kurtosis),
            "residuales_normales": bool(jb_pvalue > 0.05),
        },
    }


def run_clustering_model(
    df: pd.DataFrame, n_clusters: int = 4, output_dir: Path | None = None
) -> dict:
    """
    Opción B: K-means para segmentar clientes por comportamiento de compra.
    """
    client_features = (
        df.groupby("CODIGO CLIENTE")
        .agg(
            total_monto=("MONTO APLICADO", "sum"),
            total_unidades=("UNIDADES", "sum"),
            n_transacciones=("BOLETA", "nunique"),
            descuento_prom=("PORCENTAJE DESCUENTO", "mean"),
        )
        .dropna()
    )

    if len(client_features) < n_clusters * 2:
        return {"error": "Clientes insuficientes para clustering"}

    scaler = StandardScaler()
    X = scaler.fit_transform(client_features)

    train_idx, test_idx = train_test_split(
        range(len(X)), train_size=TRAIN_TEST_RATIO, random_state=SEED
    )
    X_train = X[train_idx]

    # Método del codo: se ajusta K-Means para k=2..8 sobre el mismo train set y
    # se grafica la inercia (WCSS) para justificar por qué se elige k=4 en vez
    # de asumirlo sin evidencia.
    elbow_inertias = _compute_elbow_curve(X_train, output_dir)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    kmeans.fit(X_train)
    labels = kmeans.predict(X)

    # silhouette_score es O(n^2) en memoria/tiempo (matriz de distancias completa);
    # con cientos de miles de clientes eso es inviable, así que se estima sobre una
    # muestra aleatoria fija (semilla CPYD_SEED) en vez de la población completa.
    sil_sample_size = min(10_000, len(X))
    sil = float(
        silhouette_score(X, labels, sample_size=sil_sample_size, random_state=SEED)
    )

    # Perfil de cada clúster: promedio de las variables originales (no
    # estandarizadas) por clúster, para poder describir qué representa cada
    # segmento (ej. "clúster de alto gasto") y no solo cuántos clientes tiene.
    profiled = client_features.copy()
    profiled["cluster"] = labels
    cluster_profile = profiled.groupby("cluster").mean().round(1).to_dict(orient="index")

    return {
        "model_type": "K-Means Clustering",
        "n_clusters": n_clusters,
        "silhouette_score": sil,
        "inertia": float(kmeans.inertia_),
        "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
        "cluster_profile": cluster_profile,
        "elbow_inertias": elbow_inertias,
    }


def _compute_elbow_curve(X_train: np.ndarray, output_dir: Path | None = None) -> dict:
    """Ajusta K-Means para k=2..8 y grafica la inercia (WCSS) resultante."""
    ks = list(range(2, 9))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_train)
        inertias.append(float(km.inertia_))

    out = output_dir or OUTPUT_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, marker="o")
    ax.set_xlabel("Número de clústeres (k)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo para elegir k")
    plt.tight_layout()
    plt.savefig(out / "clustering_metodo_codo.png", dpi=150)
    plt.close()

    return dict(zip(ks, inertias))


def _compute_vif(df: pd.DataFrame, volume_col: str = "UNIDADES") -> dict:
    """Calcula VIF para detectar multicolinealidad."""
    numeric = df[["LOCAL", volume_col, "PORCENTAJE DESCUENTO"]].dropna()
    if len(numeric) < 10:
        return {}
    vif = {}
    for i, col in enumerate(numeric.columns):
        vif[col] = float(variance_inflation_factor(numeric.values, i))
    return vif
