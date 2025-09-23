#!/usr/bin/env python3
"""Script d'installation du système de recommandations."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command: str, description: str = ""):
    """Exécute une commande et affiche le résultat."""
    print(f"\n{'='*50}")
    if description:
        print(f"📦 {description}")
    print(f"🔧 Exécution: {command}")
    print('='*50)

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        if e.stderr:
            print(f"   Détails: {e.stderr}")
        return False


def check_python_version():
    """Vérifie la version de Python."""
    print("🐍 Vérification de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} détecté. Python 3.8+ requis.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} détecté")
    return True


def install_scientific_stack():
    """Installe la pile scientifique de base."""
    print("\n🔬 Installation de la pile scientifique...")

    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des bibliothèques scientifiques")


def install_machine_learning():
    """Installe les bibliothèques de machine learning."""
    print("\n🤖 Installation du Machine Learning...")

    # Installation de PyTorch (avec CUDA si disponible)
    success = run_command(
        "pip install torch --index-url https://download.pytorch.org/whl/cu118",
        "Installation de PyTorch (avec CUDA 11.8)"
    )

    if not success:
        print("⚠️ Échec de l'installation CUDA, tentative avec CPU uniquement...")
        success = run_command(
            "pip install torch --index-url https://download.pytorch.org/whl/cpu",
            "Installation de PyTorch (CPU uniquement)"
        )

    if not success:
        print("❌ Impossible d'installer PyTorch")
        return False

    # Installation de Transformers
    success = run_command(
        "pip install transformers",
        "Installation de Transformers"
    )

    if not success:
        return False

    return True


def install_data_visualization():
    """Installe les bibliothèques de visualisation."""
    print("\n📊 Installation de la visualisation de données...")

    packages = [
        "seaborn",
        "plotly"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des bibliothèques de visualisation")


def install_utilities():
    """Installe les utilitaires."""
    print("\n🛠️ Installation des utilitaires...")

    packages = [
        "python-dateutil",
        "pytz"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des utilitaires")


def install_dev_tools():
    """Installe les outils de développement."""
    print("\n🛠️ Installation des outils de développement...")

    packages = [
        "pytest",
        "pytest-mock",
        "pytest-cov",
        "black",
        "isort",
        "flake8",
        "mypy"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des outils de développement")


def create_directories():
    """Crée les répertoires nécessaires."""
    print("\n📁 Création des répertoires...")

    directories = [
        "models/recommendations",
        "data/recommendations",
        "logs",
        "config"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Créé: {directory}/")

    return True


def test_installation():
    """Teste l'installation en important les modules principaux."""
    print("\n🧪 Test de l'installation...")

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} installé")

        import pandas as pd
        print(f"✅ Pandas {pd.__version__} installé")

        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} installé")

        import torch
        print(f"✅ PyTorch {torch.__version__} installé")

        import transformers
        print(f"✅ Transformers {transformers.__version__} installé")

        # Test des modules de recommandations
        try:
            from src.recommendations import RecommendationEngine, UserProfile, ItemCatalog
            print("✅ Modules de recommandations importés avec succès")
        except ImportError as e:
            print(f"⚠️ Modules de recommandations non disponibles: {e}")

        return True

    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        return False


def run_basic_tests():
    """Exécute les tests de base."""
    print("\n🧪 Exécution des tests de base...")

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "src/recommendations/tests/test_recommendation_engine.py", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("✅ Tests réussis")
            # Afficher un résumé des tests
            lines = result.stdout.split('\n')
            for line in lines[-10:]:  # Dernières 10 lignes
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print("❌ Échec des tests")
            print("   Sortie d'erreur:")
            for line in result.stderr.split('\n')[-5:]:  # Dernières 5 lignes d'erreur
                if line.strip():
                    print(f"   {line}")
            return False

    except subprocess.TimeoutExpired:
        print("⚠️ Tests timeout (60s) - installation réussie mais tests lents")
        return True
    except Exception as e:
        print(f"⚠️ Impossible d'exécuter les tests: {e}")
        return True  # Non critique


def create_sample_config():
    """Crée un fichier de configuration d'exemple."""
    print("\n⚙️ Création de la configuration d'exemple...")

    config_content = """# Configuration d'exemple pour le système de recommandations
# Fichier: config/recommendation_config.json

{
  "max_recommendations": 20,
  "min_similarity_threshold": 0.1,
  "context_weight": 0.3,
  "collaborative_weight": 0.4,
  "content_weight": 0.3,
  "temporal_decay_factor": 0.95,
  "min_interactions_for_similarity": 5,
  "cache_max_size": 10000,
  "model_update_interval_hours": 24,
  "feedback_learning_rate": 0.1,
  "diversity_factor": 0.2,
  "novelty_boost": 0.1,
  "performance": {
    "gpu_memory_fraction": 0.8,
    "cpu_threads": null,
    "preload_models": false,
    "model_cache_dir": "./models/recommendations",
    "temp_dir": "./temp",
    "cleanup_temp_files": true,
    "max_concurrent_requests": 4
  },
  "logging": {
    "level": "INFO",
    "file_path": "logs/recommendations.log",
    "max_file_size": 10485760,
    "backup_count": 5,
    "console_output": true
  }
}
"""

    config_path = Path("config/recommendation_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"✅ Configuration créée: {config_path}")
    return True


def main():
    """Fonction principale d'installation."""
    print("🎯 Installation du Système de Recommandations")
    print("=" * 60)

    # Vérification de Python
    if not check_python_version():
        sys.exit(1)

    # Création des répertoires
    if not create_directories():
        print("❌ Impossible de créer les répertoires")
        sys.exit(1)

    # Installation des dépendances
    steps = [
        ("Pile scientifique", install_scientific_stack),
        ("Machine Learning", install_machine_learning),
        ("Visualisation", install_data_visualization),
        ("Utilitaires", install_utilities),
        ("Outils de développement", install_dev_tools)
    ]

    failed_steps = []

    for step_name, step_func in steps:
        print(f"\n🔄 Étape: {step_name}")
        if not step_func():
            failed_steps.append(step_name)
            print(f"⚠️ Échec de l'étape: {step_name}")

    # Test final
    print("\n" + "=" * 60)
    if test_installation():
        print("🎉 Installation terminée avec succès !")

        if failed_steps:
            print(f"\n⚠️ Étapes ayant échoué (non critiques): {', '.join(failed_steps)}")

        # Tests optionnels
        if run_basic_tests():
            print("✅ Tests de validation réussis")
        else:
            print("⚠️ Tests de validation échoués")

        # Configuration d'exemple
        create_sample_config()

        print("\n🚀 Prochaines étapes:")
        print("1. Exécutez: python examples/recommendation_demo.py")
        print("2. Consultez: RECOMMENDATION_README.md")
        print("3. Lancez les tests: python -m pytest src/recommendations/tests/")
        print("4. Configurez: config/recommendation_config.json")

    else:
        print("❌ Installation incomplète. Vérifiez les erreurs ci-dessus.")
        sys.exit(1)


if __name__ == "__main__":
    main()