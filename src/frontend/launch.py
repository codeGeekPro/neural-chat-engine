#!/usr/bin/env python3
"""
Lanceur rapide pour l'interface utilisateur Neural Chat Engine

Ce script permet de lancer facilement l'interface Streamlit avec
les bonnes configurations et vérifications préalables.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    required_packages = [
        'streamlit',
        'plotly',
        'pydantic',
        'requests'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Dépendances manquantes. Installation en cours...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', *missing_packages
        ])
        print("✅ Dépendances installées!")

def check_api_availability():
    """Vérifie que l'API backend est accessible."""
    import requests

    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API backend accessible")
            return True
        else:
            print("⚠️ API backend répond mais avec un code d'erreur")
            return False
    except requests.RequestException:
        print("❌ API backend non accessible sur http://localhost:8000")
        print("   Lancez d'abord : python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
        return False

def launch_streamlit():
    """Lance l'application Streamlit."""
    print("🚀 Lancement de l'interface Streamlit...")

    # Chemin vers l'application
    app_path = Path(__file__).parent / "streamlit_app.py"

    # Commande Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--theme.base", "light"
    ]

    try:
        subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'interface utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du lancement : {e}")
        return False

    return True

def main():
    """Fonction principale."""
    print("🤖 Neural Chat Engine - Interface Utilisateur")
    print("=" * 50)

    # Vérifications préalables
    print("🔍 Vérifications préalables...")

    # Vérifier les dépendances
    check_dependencies()

    # Vérifier l'API
    if not check_api_availability():
        print("\n💡 Conseils :")
        print("   1. Lancez l'API backend dans un autre terminal")
        print("   2. Vérifiez que le port 8000 n'est pas utilisé")
        print("   3. Contrôlez les variables d'environnement dans .env")
        return

    print("\n🎯 Lancement de l'interface...")
    print("   URL: http://localhost:8501")
    print("   API: http://localhost:8000")
    print("   Appuyez sur Ctrl+C pour arrêter")
    print("-" * 50)

    # Lancer Streamlit
    success = launch_streamlit()

    if success:
        print("✅ Interface arrêtée proprement")
    else:
        print("❌ Erreur lors de l'exécution")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/workspaces/neural-chat-engine/src/frontend/launch.py