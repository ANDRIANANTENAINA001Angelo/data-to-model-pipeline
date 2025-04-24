#!/bin/bash
cd "$(dirname "$0")/../../scripts"
echo "=== Démarrage du serveur Flask..."
python3 predict.py
