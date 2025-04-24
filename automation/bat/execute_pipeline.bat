@echo off
title ⚙️ Pipeline ML - Entraînement complet
cd /d "%~dp0..\..\scripts"

echo === Nettoyage du dataset...
python cleaner.py

echo === Choix du meilleur modèle...
python choose_model.py

echo === Optimisation du modèle choisi...
python optimize.py

echo.
echo ✅ Pipeline terminé avec succès !

echo === Lancement de l'API Flask dans une nouvelle fenêtre...
start cmd /k "cd /d %~dp0..\..\scripts && title 🚀 Flask API - Predict && python predict.py"
