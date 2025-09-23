#!/usr/bin/env python3
"""Script de démarrage pour le Neural Chat Engine API."""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_dependencies():
    """Vérifie que les dépendances sont installées."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Dépendances vérifiées avec succès")
        return True
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("Installez les dépendances avec: pip install -r requirements.txt")
        return False


def create_env_file():
    """Crée un fichier .env d'exemple si nécessaire."""
    env_path = Path(".env")
    if not env_path.exists():
        env_content = """# Configuration du Neural Chat Engine
DEBUG=true
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Sécurité
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
JWT_EXPIRATION=3600

# Base de données
DATABASE_URL=sqlite:///./chat_engine.db

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Limites de taux
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# WebSocket
WS_MAX_CONNECTIONS=1000
WS_MESSAGE_TIMEOUT=30

# Uploads
UPLOAD_DIR=/tmp/uploads
MAX_UPLOAD_SIZE=10485760
"""
        env_path.write_text(env_content)
        print("📄 Fichier .env créé avec la configuration par défaut")


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True, workers: int = 1):
    """Démarre l'API FastAPI."""
    print(f"🚀 Démarrage du Neural Chat Engine sur {host}:{port}")

    # Variables d'environnement
    env = os.environ.copy()
    env.update({
        "HOST": host,
        "PORT": str(port),
        "WORKERS": str(workers),
        "DEBUG": str(reload).lower()
    })

    # Commande uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers)
    ]

    if reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du serveur")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        sys.exit(1)


def run_tests():
    """Exécute les tests."""
    print("🧪 Exécution des tests...")
    try:
        subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--cov=src",
            "--cov-report=html"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec des tests: {e}")
        sys.exit(1)


def format_code():
    """Formate le code avec black et isort."""
    print("🎨 Formatage du code...")
    try:
        subprocess.run([sys.executable, "-m", "black", "src/"], check=True)
        subprocess.run([sys.executable, "-m", "isort", "src/"], check=True)
        print("✅ Code formaté")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur de formatage: {e}")
        sys.exit(1)


def lint_code():
    """Vérifie le code avec flake8 et mypy."""
    print("🔍 Vérification du code...")
    try:
        subprocess.run([sys.executable, "-m", "flake8", "src/"], check=True)
        subprocess.run([sys.executable, "-m", "mypy", "src/"], check=True)
        print("✅ Code vérifié")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreurs de linting: {e}")
        sys.exit(1)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Neural Chat Engine - Outil de gestion")
    parser.add_argument("command", choices=[
        "start", "test", "format", "lint", "check-deps", "create-env"
    ], help="Commande à exécuter")

    parser.add_argument("--host", default="0.0.0.0", help="Hôte pour le serveur (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port pour le serveur (défaut: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Désactiver le rechargement automatique")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de workers (défaut: 1)")

    args = parser.parse_args()

    if args.command == "check-deps":
        if not check_dependencies():
            sys.exit(1)

    elif args.command == "create-env":
        create_env_file()

    elif args.command == "start":
        if not check_dependencies():
            sys.exit(1)
        create_env_file()
        start_api(
            host=args.host,
            port=args.port,
            reload=not args.no_reload,
            workers=args.workers
        )

    elif args.command == "test":
        run_tests()

    elif args.command == "format":
        format_code()

    elif args.command == "lint":
        lint_code()


if __name__ == "__main__":
    main()