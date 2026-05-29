---
title: Fruit Cluster AI
emoji: 🍎
colorFrom: blue
colorTo: orange
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

# Fruit Clustering — Apprentissage non supervisé + XAI (SHAP)

Application de démonstration du pipeline MLOps complet :

**Dataset :** fruits.csv — 400 observations, 2 features  
**Modèle :** K-Means (k=3) — Silhouette Score : 0.70  
**Explainability :** SHAP via modèle proxy Random Forest  

## Utilisation

1. Entrez les valeurs de `feature_1` et `feature_2`
2. Cliquez sur **Prédire et expliquer**
3. Visualisez le cluster assigné et les explications SHAP

## Stack technique

- scikit-learn (K-Means, CAH, DBSCAN, Random Forest)
- SHAP (TreeExplainer)
- MLflow (tracking)
- DVC (versionnage données)
- GitHub Actions (CI/CD)
- HuggingFace Spaces (déploiement)
