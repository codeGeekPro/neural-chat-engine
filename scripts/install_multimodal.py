#!/usr/bin/env python3
"""Script d'installation des capacités multimodales."""

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


def install_core_dependencies():
    """Installe les dépendances de base."""
    print("\n🚀 Installation des dépendances de base...")

    # Installation de PyTorch (avec CUDA si disponible)
    success = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Installation de PyTorch (avec CUDA 11.8)"
    )

    if not success:
        print("⚠️ Échec de l'installation CUDA, tentative avec CPU uniquement...")
        success = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "Installation de PyTorch (CPU uniquement)"
        )

    if not success:
        print("❌ Impossible d'installer PyTorch")
        return False

    # Installation des transformers
    success = run_command(
        "pip install transformers accelerate",
        "Installation de Transformers"
    )

    if not success:
        return False

    return True


def install_vision_dependencies():
    """Installe les dépendances pour le traitement d'images."""
    print("\n🖼️ Installation des dépendances vision...")

    packages = [
        "Pillow",
        "opencv-python",
        "scikit-image",
        "matplotlib"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des bibliothèques de vision")


def install_audio_dependencies():
    """Installe les dépendances pour le traitement audio."""
    print("\n🎵 Installation des dépendances audio...")

    packages = [
        "librosa",
        "soundfile",
        "pyaudio"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des bibliothèques audio")


def install_ocr_dependencies():
    """Installe les dépendances OCR (optionnel)."""
    print("\n📝 Installation d'EasyOCR (optionnel)...")

    try:
        success = run_command("pip install easyocr", "Installation d'EasyOCR")
        if success:
            print("✅ OCR installé avec succès")
        else:
            print("⚠️ OCR non installé (optionnel)")
        return True
    except Exception as e:
        print(f"⚠️ Impossible d'installer OCR: {e}")
        return True  # Non critique


def install_tts_dependencies():
    """Installe les dépendances TTS."""
    print("\n🗣️ Installation des dépendances TTS...")

    packages = [
        "pyttsx3",
        "gTTS"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des bibliothèques TTS")


def install_scientific_dependencies():
    """Installe les dépendances scientifiques."""
    print("\n🔬 Installation des dépendances scientifiques...")

    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des bibliothèques scientifiques")


def install_dev_dependencies():
    """Installe les dépendances de développement."""
    print("\n🛠️ Installation des dépendances de développement...")

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


def install_config_dependencies():
    """Installe les dépendances de configuration."""
    print("\n⚙️ Installation des dépendances de configuration...")

    packages = [
        "pyyaml",
        "loguru"
    ]

    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installation des bibliothèques de configuration")


def create_directories():
    """Crée les répertoires nécessaires."""
    print("\n📁 Création des répertoires...")

    directories = [
        "models",
        "temp",
        "logs",
        "config",
        "data",
        "examples"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Créé: {directory}/")

    return True


def test_installation():
    """Teste l'installation en important les modules principaux."""
    print("\n🧪 Test de l'installation...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installé")

        import transformers
        print(f"✅ Transformers {transformers.__version__} installé")

        import PIL
        print(f"✅ Pillow {PIL.__version__} installé")

        import librosa
        print(f"✅ Librosa {librosa.__version__} installé")

        import numpy as np
        print(f"✅ NumPy {np.__version__} installé")

        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} installé")

        # Test des processeurs multimodaux
        try:
            from src.multimodal import VisionProcessor, AudioProcessor
            print("✅ Processeurs multimodaux importés avec succès")
        except ImportError as e:
            print(f"⚠️ Processeurs non disponibles: {e}")

        return True

    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        return False


def main():
    """Fonction principale d'installation."""
    print("🎯 Installation des Capacités Multimodales")
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
        ("Dépendances de base", install_core_dependencies),
        ("Vision", install_vision_dependencies),
        ("Audio", install_audio_dependencies),
        ("OCR", install_ocr_dependencies),
        ("TTS", install_tts_dependencies),
        ("Scientifique", install_scientific_dependencies),
        ("Configuration", install_config_dependencies),
        ("Développement", install_dev_dependencies)
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

        print("\n🚀 Prochaines étapes:")
        print("1. Exécutez: python examples/multimodal_demo.py")
        print("2. Consultez: MULTIMODAL_README.md")
        print("3. Lancez les tests: python -m pytest src/multimodal/tests/")

    else:
        print("❌ Installation incomplète. Vérifiez les erreurs ci-dessus.")
        sys.exit(1)


if __name__ == "__main__":
    main()