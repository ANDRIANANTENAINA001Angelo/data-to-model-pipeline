from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
import pandas as pd
import joblib
import yaml
import numpy as np
import os

# === Initialisation de l'app ===
app = Flask(__name__)
swagger = Swagger(app)

# === Chargement de la configuration ===
# Récupère le chemin absolu du fichier YAML par rapport à ce script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../config/pipeline_config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PIPELINE_PATH = os.path.join(BASE_DIR, "../models/trained/full_pipeline.pkl")

# === Chargement du pipeline complet déjà entraîné ===
pipeline = joblib.load(PIPELINE_PATH)

# === Endpoint de prédiction ===
@app.route("/predict", methods=["POST"])
@swag_from({
    'summary': 'Prédiction du salaire en USD',
    'description': "Envoie les données d’un poste pour prédire le salaire en USD.",
    'consumes': ['application/json'],
    'produces': ['application/json'],
    'parameters': [{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            'type': 'object',
            'example': {
                "work_year": 2025,
                "experience_level": "SE",
                "employment_type": "FT",
                "job_title": "Data Scientist",
                "company_size": "M",
                "remote_ratio": 100,
                "employee_residence": "US",
                "company_location": "US"
            }
        }
    }],
    'responses': {
        200: {
            'description': 'Succès - Prédiction retournée',
            'schema': {
                'type': 'object',
                'example': {
                    'prediction': 135000.0
                }
            }
        },
        400: {
            'description': 'Requête invalide (pas de données)',
            'schema': {
                'type': 'object',
                'example': {
                    'error': 'Aucune donnée reçue'
                }
            }
        },
        500: {
            'description': 'Erreur serveur (ex: problème lors de la prédiction)',
            'schema': {
                'type': 'object',
                'example': {
                    'error': 'some internal error message'
                }
            }
        }
    }
})
def predict():
    input_data = request.get_json()

    if not input_data:
        return jsonify({"error": "Aucune donnée reçue"}), 400

    try:
        df_input = pd.DataFrame([input_data])
        prediction = pipeline.predict(df_input)[0]
        return jsonify({"prediction": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Lancement de l'app Flask ===
if __name__ == "__main__":
    app.run(debug=True)
