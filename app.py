"""
app.py — Interface Gradio pour HuggingFace Spaces
Clustering de fruits + Explainability SHAP
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import gradio as gr

# ── Chargement des modèles ──────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

kmeans   = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
rf_proxy = joblib.load(os.path.join(MODEL_DIR, "rf_proxy.pkl"))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    metadata = json.load(f)

best_k        = metadata["best_k"]
feature_names = metadata["feature_names"]
cluster_info  = metadata["cluster_info"]

PALETTE = ['#4C72B0', '#DD8452', '#2ca02c', '#d62728', '#9467bd']

explainer = shap.TreeExplainer(rf_proxy)


# ── Helpers ─────────────────────────────────────────────────────
def predict_cluster(f1: float, f2: float):
    """Prédit le cluster et calcule les SHAP values pour un fruit."""
    X_raw  = np.array([[f1, f2]])
    X_norm = scaler.transform(X_raw)
    cluster = int(kmeans.predict(X_norm)[0])

    # SHAP
    sv = explainer.shap_values(X_norm)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    # sv shape: (1, 2, k)
    sv_cluster = sv[0, :, cluster]

    return cluster, X_norm[0], sv_cluster


def make_scatter_fig(highlight_point=None, highlight_cluster=None):
    """Nuage de points du dataset avec les clusters K-Means."""
    # Recharger toutes les données normalisées pour le plot
    try:
        df_all = pd.read_csv(os.path.join(BASE_DIR, "data", "fruits.csv"),
                             header=None, names=["f1", "f2"])
        X_all  = scaler.transform(df_all.values)
        labels = kmeans.predict(X_all)
    except Exception:
        X_all  = None
        labels = None

    fig, ax = plt.subplots(figsize=(6, 5))

    if X_all is not None:
        for c in range(best_k):
            mask = labels == c
            ax.scatter(X_all[mask, 0], X_all[mask, 1],
                       c=PALETTE[c % len(PALETTE)],
                       label=f"Cluster {c}  ({mask.sum()} pts)",
                       alpha=0.5, s=25, edgecolors="none")

    if highlight_point is not None:
        ax.scatter(highlight_point[0], highlight_point[1],
                   c=PALETTE[highlight_cluster % len(PALETTE)],
                   s=250, marker="*", edgecolors="black", linewidths=1.5,
                   zorder=10, label="Nouveau fruit")

    ax.set_title("Dataset — clusters K-Means")
    ax.set_xlabel("feature_1 (normalisée)")
    ax.set_ylabel("feature_2 (normalisée)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def make_shap_fig(sv_cluster, cluster_id, f1_norm, f2_norm):
    """Waterfall SHAP pour un fruit individuel."""
    ev = explainer.expected_value
    base_val = ev[cluster_id] if hasattr(ev, "__len__") else ev

    explanation = shap.Explanation(
        values=sv_cluster,
        base_values=base_val,
        data=np.array([f1_norm, f2_norm]),
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    shap.waterfall_plot(explanation, show=False)
    plt.title(f"SHAP — Pourquoi Cluster {cluster_id} ?", fontsize=12)
    plt.tight_layout()
    return fig


def make_global_importance_fig():
    """Importance SHAP globale (précalculée via metadata)."""
    shap_imp = metadata["shap_importance"]
    names    = list(shap_imp.keys())
    vals     = list(shap_imp.values())

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(names, vals, color=["#4C72B0", "#DD8452"], edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.set_title("Importance SHAP globale")
    ax.set_xlabel("|SHAP value| moyenne")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ── Fonction principale Gradio ───────────────────────────────────
def predict_and_explain(f1, f2):
    cluster, X_norm, sv = predict_cluster(f1, f2)

    # Infos cluster
    info = cluster_info.get(str(cluster), {})
    txt  = (
        f"**Cluster assigné : {cluster}**\n\n"
        f"- Taille du cluster : {info.get('size', '?')} fruits\n"
        f"- Centroïde feature_1 : {info.get('centroid_f1', 0):.2f}\n"
        f"- Centroïde feature_2 : {info.get('centroid_f2', 0):.2f}\n\n"
        f"**Explication SHAP :**\n"
        f"- feature_1 contribue : {sv[0]:+.4f}\n"
        f"- feature_2 contribue : {sv[1]:+.4f}\n\n"
        f"*Valeur positive = pousse vers ce cluster*  \n"
        f"*Valeur négative = éloigne de ce cluster*"
    )

    fig_scatter  = make_scatter_fig(highlight_point=X_norm, highlight_cluster=cluster)
    fig_shap     = make_shap_fig(sv, cluster, X_norm[0], X_norm[1])
    fig_global   = make_global_importance_fig()

    return txt, fig_scatter, fig_shap, fig_global


# ── Interface Gradio ─────────────────────────────────────────────
with gr.Blocks(title="Fruit Clustering + SHAP", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Fruit Clustering — Apprentissage non supervisé + XAI (SHAP)
    Entrez les caractéristiques d'un fruit pour connaître son cluster et comprendre pourquoi.
    """)

    with gr.Row():
        f1_input = gr.Slider(minimum=0, maximum=80, value=25.0, step=0.5,
                             label="feature_1")
        f2_input = gr.Slider(minimum=0, maximum=100, value=8.0, step=0.5,
                             label="feature_2")

    btn = gr.Button("Prédire et expliquer", variant="primary")

    with gr.Row():
        result_text = gr.Markdown(label="Résultat")

    with gr.Row():
        plot_scatter = gr.Plot(label="Position dans le dataset")
        plot_shap    = gr.Plot(label="Explication SHAP (waterfall)")

    plot_global = gr.Plot(label="Importance SHAP globale")

    btn.click(
        fn=predict_and_explain,
        inputs=[f1_input, f2_input],
        outputs=[result_text, plot_scatter, plot_shap, plot_global]
    )

    gr.Markdown("""
    ---
    **Méthodologie :**
    K-Means → pseudo-labels → Random Forest proxy → SHAP values  
    Silhouette Score K-Means : **{:.4f}** | k = {}
    """.format(metadata["silhouette_kmeans"], best_k))


if __name__ == "__main__":
    demo.launch()
