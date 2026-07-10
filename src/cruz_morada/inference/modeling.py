"""Modelado predictivo y descriptivo."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from cruz_morada.config import SEED, TRAIN_TEST_RATIO

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

    return {
        "model_type": "OLS Regression",
        "r_squared_adj": float(model.rsquared_adj),
        "rmse_test": rmse,
        "mae_test": mae,
        "coefficients": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "vif": vif_data,
        "summary_excerpt": str(model.summary().tables[1]),
    }


def run_clustering_model(df: pd.DataFrame, n_clusters: int = 4) -> dict:
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

    return {
        "model_type": "K-Means Clustering",
        "n_clusters": n_clusters,
        "silhouette_score": sil,
        "inertia": float(kmeans.inertia_),
        "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
    }


def _compute_vif(df: pd.DataFrame, volume_col: str = "UNIDADES") -> dict:
    """Calcula VIF para detectar multicolinealidad."""
    numeric = df[["LOCAL", volume_col, "PORCENTAJE DESCUENTO"]].dropna()
    if len(numeric) < 10:
        return {}
    vif = {}
    for i, col in enumerate(numeric.columns):
        vif[col] = float(variance_inflation_factor(numeric.values, i))
    return vif
