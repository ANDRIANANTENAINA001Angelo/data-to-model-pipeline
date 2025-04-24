#!/bin/bash
cd "$(dirname "$0")/../../scripts"

echo "=== Nettoyage du dataset..."
python3 cleaner.py

echo "=== Choix du meilleur modèle..."
python3 choose_model.py

echo "=== Optimisation du modèle choisi..."
python3 optimize.py

echo ""
echo "✅ Pipeline terminé avec succès !"

echo "=== Lancement de l'API Flask dans un terminal séparé..."
gnome-terminal -- bash -c "cd $(pwd); echo '🚀 Flask API - Predict'; python3 predict.py; exec bash"
