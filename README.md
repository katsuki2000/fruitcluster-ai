---
title: Fruitcluster Ai
emoji: 🍇
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# FruitCluster AI 

Clustering non supervisé de fruits, explicabilité (SHAP) et déploiement MLOps, exposés par une API FastAPI.

## Objectif

Découvrir des groupes naturels dans 400 fruits décrits par 2 variables numériques, sans labels, puis expliquer l'appartenance de chaque point à son cluster.

## Résultats

| Algorithme | Clusters | Silhouette |
|---|---|---|
| **K-Means** (retenu) | 3 | **0,7011** |
| CAH (Ward) | 3 | 0,7011 |
| DBSCAN (eps = 0,2) | 5 | 0,6701 |

k = 3 est choisi par la méthode du coude et le score de silhouette. Les variables sont standardisées (`StandardScaler`).

## Explicabilité (XAI)

SHAP est calculé sur un Random Forest proxy entraîné à reproduire les clusters de K-Means. Les explications portent donc sur le proxy, pas directement sur K-Means.

## Pipeline MLOps

| Étape | Outil |
|---|---|
| Modèles | scikit-learn |
| Suivi des expériences | MLflow |
| Versionnage des données | DVC |
| API | FastAPI |
| Conteneurisation / déploiement | Docker, Hugging Face Spaces |
| CI | GitHub Actions |
| Surveillance de la dérive | Evidently AI |

## Utiliser l'API

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Statut de l'API |
| POST | `/predict` | Cluster d'un fruit |
| GET | `/xai/global-importance` | Importance globale des variables (graphique SHAP) |
| GET | `/docs` | Documentation interactive (Swagger) |

Exemple :

```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{"feature_1": 41.86, "feature_2": 40.73}'
```

Réponse : le numéro de cluster, ses caractéristiques moyennes et le score de silhouette du modèle.

## Lancer en local

```bash
pip install -r requirements.txt
uvicorn api:app --port 7860
```

Ou avec Docker :

```bash
docker build -t fruitcluster-ai .
docker run -p 7860:7860 fruitcluster-ai
```

## Structure du dépôt

```
├── api.py              # API FastAPI
├── Dockerfile
├── requirements.txt
├── models/             # kmeans.pkl, scaler.pkl, metadata.json
├── data/               # fruits.csv (sans en-tête : feature_1, feature_2)
├── notebook/           # notebook d'analyse complet (Colab)
└── .github/workflows/  # CI
```


