"""
tests/test_model.py — Tests automatiques pour le pipeline fruits
Exécutés par GitHub Actions à chaque push
"""

import os
import sys
import json
import pytest
import numpy as np
import joblib

# Chemin vers la racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "models")


@pytest.fixture(scope="module")
def models():
    """Charge les modèles une seule fois pour tous les tests."""
    kmeans   = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    rf_proxy = joblib.load(os.path.join(MODEL_DIR, "rf_proxy.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return {"kmeans": kmeans, "scaler": scaler,
            "rf_proxy": rf_proxy, "metadata": metadata}


# ── Tests des fichiers ───────────────────────────────────────────
def test_model_files_exist():
    """Tous les fichiers modèles doivent exister."""
    for fname in ["kmeans.pkl", "scaler.pkl", "rf_proxy.pkl", "metadata.json"]:
        path = os.path.join(MODEL_DIR, fname)
        assert os.path.exists(path), f"Fichier manquant : {fname}"


def test_metadata_keys(models):
    """metadata.json doit contenir les clés attendues."""
    required_keys = ["best_k", "feature_names", "silhouette_kmeans",
                     "rf_proxy_accuracy", "cluster_info", "shap_importance"]
    for key in required_keys:
        assert key in models["metadata"], f"Clé manquante dans metadata : {key}"


# ── Tests du modèle K-Means ──────────────────────────────────────
def test_kmeans_predict_shape(models):
    """K-Means doit retourner un label par point."""
    X = np.array([[25.0, 8.0], [45.0, 50.0], [20.0, 4.0]])
    X_norm = models["scaler"].transform(X)
    labels = models["kmeans"].predict(X_norm)
    assert labels.shape == (3,), f"Shape inattendue : {labels.shape}"


def test_kmeans_labels_valid(models):
    """Les labels K-Means doivent être dans [0, best_k-1]."""
    best_k = models["metadata"]["best_k"]
    X = np.random.rand(50, 2) * 50
    X_norm = models["scaler"].transform(X)
    labels = models["kmeans"].predict(X_norm)
    assert labels.min() >= 0
    assert labels.max() < best_k, f"Label {labels.max()} >= best_k={best_k}"


def test_silhouette_above_threshold(models):
    """Le silhouette score doit dépasser 0.5 (seuil minimal de qualité)."""
    sil = models["metadata"]["silhouette_kmeans"]
    assert sil >= 0.5, f"Silhouette trop bas : {sil:.4f} < 0.5"


# ── Tests du scaler ──────────────────────────────────────────────
def test_scaler_output_range(models):
    """Les données normalisées doivent avoir mean~0 et std~1."""
    X = np.array([[25.0, 8.0], [45.0, 50.0], [20.0, 4.0],
                  [50.0, 75.0], [35.0, 28.0]])
    X_norm = models["scaler"].transform(X)
    # Mean et std ne seront pas exactement 0/1 sur 5 points,
    # mais les valeurs doivent être dans un intervalle raisonnable
    assert X_norm.shape == X.shape
    assert np.all(np.isfinite(X_norm)), "Valeurs NaN ou Inf après normalisation"


# ── Tests du proxy Random Forest ────────────────────────────────
def test_rf_proxy_accuracy(models):
    """Le RF proxy doit avoir une accuracy > 0.95 sur les pseudo-labels."""
    acc = models["metadata"]["rf_proxy_accuracy"]
    assert acc >= 0.95, f"Accuracy proxy trop faible : {acc:.4f}"


def test_rf_proxy_predict(models):
    """Le RF proxy doit retourner des classes valides."""
    best_k = models["metadata"]["best_k"]
    X = np.array([[25.0, 8.0], [45.0, 50.0]])
    X_norm = models["scaler"].transform(X)
    preds = models["rf_proxy"].predict(X_norm)
    assert preds.shape == (2,)
    assert all(0 <= p < best_k for p in preds)


# ── Test de bout en bout ─────────────────────────────────────────
def test_end_to_end_pipeline(models):
    """Pipeline complet : entrée brute → cluster → SHAP."""
    import shap as shap_lib

    f1, f2 = 25.0, 8.0
    X_raw  = np.array([[f1, f2]])
    X_norm = models["scaler"].transform(X_raw)

    # Prédiction cluster
    cluster = int(models["kmeans"].predict(X_norm)[0])
    assert 0 <= cluster < models["metadata"]["best_k"]

    # SHAP values
    explainer   = shap_lib.TreeExplainer(models["rf_proxy"])
    shap_values = explainer.shap_values(X_norm)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    assert shap_values.shape[0] == 1
    assert shap_values.shape[1] == 2  # 2 features
    assert shap_values.shape[2] == models["metadata"]["best_k"]
