import pandas as pd
import numpy as np
import os
import yaml
import argparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import warnings

# === Détection du chemin absolu du projet ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../config/pipeline_config.yaml")
CLEAN_OUTPUT_DIR = os.path.join(BASE_DIR, "../datasets/clean")

# === Chargement des paramètres ===
def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === Nettoyage et encodage ===
def clean_dataset(config):
    input_path = os.path.join(BASE_DIR, f"../{config['input_dataset']}")
    target = config["target"]
    cat_features = config["categorical_features"]
    exclude = config.get("exclude_columns", [])

    df = pd.read_csv(input_path)
    print(f"Dataset original shape: {df.shape}")

    df = df.drop(columns=exclude, errors='ignore')
    df.dropna(inplace=True)

    y = df[target]
    X = df.drop(columns=[target])

    text_features = [col for col in X.select_dtypes(include="object").columns if col not in cat_features]
    num_features = X.select_dtypes(include=np.number).columns.tolist()

    # Pipelines
    transformers = []
    col_names = []

    sklearn_version = sklearn.__version__.split(".")
    use_sparse_output = int(sklearn_version[0]) > 1 or (int(sklearn_version[0]) == 1 and int(sklearn_version[1]) >= 2)
    cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if use_sparse_output else OneHotEncoder(handle_unknown="ignore", sparse=False)

    if cat_features:
        cat_encoded = cat_enc.fit_transform(X[cat_features])
        cat_columns = cat_enc.get_feature_names_out(cat_features)
        transformers.append(pd.DataFrame(cat_encoded, columns=cat_columns))
        col_names.extend(cat_columns)

    if text_features:
        tfidf = TfidfVectorizer()
        tfidf_encoded = tfidf.fit_transform(X[text_features[0]]).toarray()
        tfidf_columns = [f"{text_features[0]}_{i}" for i in range(tfidf_encoded.shape[1])]
        transformers.append(pd.DataFrame(tfidf_encoded, columns=tfidf_columns))
        col_names.extend(tfidf_columns)

    if num_features:
        scaler = StandardScaler()
        num_encoded = scaler.fit_transform(X[num_features])
        transformers.append(pd.DataFrame(num_encoded, columns=num_features))
        col_names.extend(num_features)

    X_encoded = pd.concat(transformers, axis=1)
    df_clean = X_encoded.copy()
    df_clean[target] = y.reset_index(drop=True)

    os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(CLEAN_OUTPUT_DIR, "data_clean.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Dataset clean sauvegardé dans : {output_path}")

if __name__ == "__main__":
    config = load_config()
    clean_dataset(config)
