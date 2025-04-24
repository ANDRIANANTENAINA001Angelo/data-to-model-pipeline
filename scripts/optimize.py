import pandas as pd
import os
import joblib
import json
import yaml

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Répertoires
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../config/pipeline_config.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "../models/chooses/best_model.pkl")
SAVE_PATH = os.path.join(BASE_DIR, "../models/trained")
RAW_DATA_PATH = None

os.makedirs(SAVE_PATH, exist_ok=True)

# Chargement config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

target = config["target"]
RAW_DATA_PATH = os.path.join(BASE_DIR, f"../{config['input_dataset']}")

# Dataset brut
df = pd.read_csv(RAW_DATA_PATH)
df = df.drop(columns=config.get("exclude_columns", []), errors="ignore").dropna()
X = df.drop(columns=[target])
y = df[target]

# Chargement du meilleur modèle
model = joblib.load(MODEL_PATH)
model_class_name = model.__class__.__name__

# Hyperparamètres
param_grid = {}
if model_class_name == "RandomForestRegressor":
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }
elif model_class_name == "GradientBoostingRegressor":
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }
elif model_class_name == "KNeighborsRegressor":
    param_grid = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    }

# GridSearch
if param_grid:
    print(f"🔍 Optimisation du modèle : {model_class_name}")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"✅ Meilleur modèle trouvé : {best_model}")
    print(f"📈 Meilleur score : {-grid_search.best_score_:.4f}")
else:
    print("ℹ️ Pas de paramètres à optimiser. Utilisation du modèle existant.")
    best_model = model

# Reconstruction du pipeline complet
transformers = []
if config["categorical_features"]:
    transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config["categorical_features"]))

text_features = [col for col in X.select_dtypes(include="object").columns if col not in config["categorical_features"]]
if text_features:
    transformers.append(("text", TfidfVectorizer(), text_features[0]))

num_features = X.select_dtypes(include="number").columns.tolist()
if num_features:
    transformers.append(("num", StandardScaler(), num_features))

preprocessor = ColumnTransformer(transformers=transformers)
pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
pipeline.fit(X, y)

joblib.dump(pipeline, os.path.join(SAVE_PATH, "full_pipeline.pkl"))
print(f"💾 Pipeline complet sauvegardé : {SAVE_PATH}/full_pipeline.pkl")

# Sauvegarde des meilleurs paramètres
if param_grid:
    with open(os.path.join(SAVE_PATH, "best_params.json"), "w") as f:
        json.dump(grid_search.best_params_, f, indent=2)
    print(f"⚙️ Hyperparamètres sauvegardés : {SAVE_PATH}/best_params.json")
