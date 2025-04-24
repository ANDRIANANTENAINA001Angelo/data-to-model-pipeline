import pandas as pd
import numpy as np
import os
import json
import joblib
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

# === Répertoires
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../config/pipeline_config.yaml")
DATA_PATH = os.path.join(BASE_DIR, "../datasets/clean/data_clean.csv")
SAVE_PATH = os.path.join(BASE_DIR, "../models/chooses")
os.makedirs(SAVE_PATH, exist_ok=True)

# === Chargement de la config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

target = config["target"]

# Chargement du dataset clean
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[target])
y = df[target]

# Liste des modèles à tester
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor()
}

results = {}
for name, model in models.items():
    print(f"🔄 Entraînement du modèle : {name}")
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    mean_score = np.abs(scores.mean())
    results[name] = mean_score
    print(f"    → RMSE moyen : {mean_score:.4f}")

best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Meilleur modèle : {best_model_name} (RMSE moyen : {results[best_model_name]:.4f})")

best_model.fit(X, y)
joblib.dump(best_model, os.path.join(SAVE_PATH, "best_model.pkl"))
print(f"💾 Modèle sauvegardé : {SAVE_PATH}/best_model.pkl")

with open(os.path.join(SAVE_PATH, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"📊 Résultats enregistrés : {SAVE_PATH}/results.json")
