from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

app = FastAPI(
    title="FruitCluster API",
    description="API de clustering de fruits avec modèle K-Means",
    version="1.0.0"
)

kmeans = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

class FruitInput(BaseModel):
    feature_1: float
    feature_2: float

@app.get("/")
def home():
    return {
        "message": "FruitCluster API fonctionne",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: FruitInput):
    X = np.array([[data.feature_1, data.feature_2]])
    X_scaled = scaler.transform(X)
    cluster = int(kmeans.predict(X_scaled)[0])

    return {
        "feature_1": data.feature_1,
        "feature_2": data.feature_2,
        "cluster": cluster,
        "cluster_info": metadata["cluster_info"].get(str(cluster)),
        "silhouette_kmeans": metadata["silhouette_kmeans"],
        "best_k": metadata["best_k"]
    }
