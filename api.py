from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

app = FastAPI(
    title="FruitCluster API + XAI",
    description="API de clustering de fruits avec explications SHAP globales",
    version="1.1.0"
)

# CORRECTION : On ne charge que les deux artefacts indispensables
kmeans = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

class FruitInput(BaseModel):
    feature_1: float
    feature_2: float

@app.get("/", tags=["General"])
def home():
    return {"message": "FruitCluster API fonctionnelle. Allez sur /docs pour le XAI."}

@app.post("/predict", tags=["Predictions"])
def predict(data: FruitInput):
    X_raw = np.array([[data.feature_1, data.feature_2]])
    X_scaled = scaler.transform(X_raw)
    cluster = int(kmeans.predict(X_scaled)[0])

    return {
        "feature_1": data.feature_1,
        "feature_2": data.feature_2,
        "cluster": cluster,
        "cluster_info": metadata["cluster_info"].get(str(cluster)),
        "silhouette_kmeans": metadata["silhouette_kmeans"]
    }

@app.get("/xai/global-importance", tags=["Explainability (XAI)"])
def get_global_shap_plot():
    img_path = "global_shap_importance.png"
    
    importances = metadata["shap_importance"]
    features = list(importances.keys())
    values = list(importances.values())
    
    plt.figure(figsize=(6, 3.5))
    plt.barh(features, values, color=['#4C72B0', '#DD8452'], alpha=0.85, edgecolor='white')
    plt.xlabel("|SHAP value| moyenne cumuler")
    plt.title("Importance Globale des Variables (Modèle Proxy)")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(img_path, dpi=120)
    plt.close()
    
    return FileResponse(img_path, media_type="image/png")
