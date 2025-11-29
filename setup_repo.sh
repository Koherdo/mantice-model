#!/bin/bash
# Script de configuration du repository Mantice

echo "Création de la structure du repository..."

# Créer les répertoires
mkdir -p .github/workflows
mkdir -p src
mkdir -p tests
mkdir -p examples
mkdir -p figures
mkdir -p docs
mkdir -p results

# Créer les fichiers principaux
touch .github/workflows/ci.yml

touch src/__init__.py
touch src/quaternions.py
touch src/primatron.py
touch src/synchronization.py
touch src/mantice_detection.py
touch src/visualization.py
touch src/cli.py

touch tests/__init__.py
touch tests/test_quaternions.py
touch tests/test_primatron.py

touch examples/turbulence_simulation.py
touch examples/railway_optimization.py

touch figures/phase_transition.py
touch figures/transport_scaling.py
touch figures/energy_spectrum.py
touch figures/railway_recovery.py

touch docs/api.md
touch docs/theory.md
touch docs/tutorials.md

touch .gitignore
touch CITATION.cff
touch CODE_OF_CONDUCT.md
touch environment.yml
touch LICENSE
touch README.md
touch reproduce_paper.py
touch setup.py
touch requirements.txt

echo "Structure créée avec succès!"
echo ""
echo "Arborescence créée :"
find . -type d -not -path "./.git/*" | sort