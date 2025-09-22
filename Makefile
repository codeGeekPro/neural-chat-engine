# ====================
# Neural Chat Engine - Makefile
# Commandes utiles pour le développement et déploiement
# ====================

.PHONY: help install test clean lint format docker run-dev run-prod config-test

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := neural-chat-engine
DOCKER_IMAGE := $(PROJECT_NAME):latest
VENV_DIR := venv

# Colors pour l'affichage
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Affiche l'aide
	@echo "$(GREEN)Neural Chat Engine - Commandes disponibles:$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installe les dépendances
	@echo "$(YELLOW)Installation des dépendances...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)✅ Installation terminée!$(NC)"

install-dev: ## Installe les dépendances de développement
	@echo "$(YELLOW)Installation des dépendances de développement...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)✅ Installation dev terminée!$(NC)"

venv: ## Crée un environnement virtuel
	@echo "$(YELLOW)Création de l'environnement virtuel...$(NC)"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(GREEN)✅ Environnement virtuel créé dans $(VENV_DIR)$(NC)"
	@echo "$(YELLOW)Activez-le avec: source $(VENV_DIR)/bin/activate$(NC)"

test: ## Lance les tests
	@echo "$(YELLOW)Lancement des tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html
	@echo "$(GREEN)✅ Tests terminés!$(NC)"

config-test: ## Test la configuration
	@echo "$(YELLOW)Test du système de configuration...$(NC)"
	$(PYTHON) test_config.py
	@echo "$(GREEN)✅ Test de configuration terminé!$(NC)"

lint: ## Vérifie la qualité du code
	@echo "$(YELLOW)Vérification de la qualité du code...$(NC)"
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✅ Linting terminé!$(NC)"

format: ## Formate le code
	@echo "$(YELLOW)Formatage du code...$(NC)"
	black src/ tests/ --line-length=88
	isort src/ tests/ --profile=black
	@echo "$(GREEN)✅ Formatage terminé!$(NC)"

clean: ## Nettoie les fichiers temporaires
	@echo "$(YELLOW)Nettoyage...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	@echo "$(GREEN)✅ Nettoyage terminé!$(NC)"

docker-build: ## Construit l'image Docker
	@echo "$(YELLOW)Construction de l'image Docker...$(NC)"
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✅ Image Docker construite: $(DOCKER_IMAGE)$(NC)"

docker-run: ## Lance le conteneur Docker
	@echo "$(YELLOW)Lancement du conteneur Docker...$(NC)"
	docker run -p 8000:8000 -p 8501:8501 --name $(PROJECT_NAME)-container $(DOCKER_IMAGE)

docker-dev: ## Lance l'environnement de développement avec Docker Compose
	@echo "$(YELLOW)Lancement de l'environnement de développement...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Environnement de développement lancé!$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Frontend: http://localhost:8501$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3000$(NC)"

docker-prod: ## Lance l'environnement de production
	@echo "$(YELLOW)Lancement de l'environnement de production...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "$(GREEN)✅ Environnement de production lancé!$(NC)"

docker-stop: ## Arrête tous les conteneurs
	@echo "$(YELLOW)Arrêt des conteneurs...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Conteneurs arrêtés!$(NC)"

docker-logs: ## Affiche les logs des conteneurs
	docker-compose logs -f

run-api: ## Lance l'API en mode développement
	@echo "$(YELLOW)Lancement de l'API en mode développement...$(NC)"
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-frontend: ## Lance le frontend Streamlit
	@echo "$(YELLOW)Lancement du frontend Streamlit...$(NC)"
	streamlit run src/frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

run-notebook: ## Lance Jupyter Lab
	@echo "$(YELLOW)Lancement de Jupyter Lab...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

celery-worker: ## Lance un worker Celery
	@echo "$(YELLOW)Lancement du worker Celery...$(NC)"
	celery -A src.api.celery_app worker --loglevel=info

celery-flower: ## Lance le monitoring Celery Flower
	@echo "$(YELLOW)Lancement de Celery Flower...$(NC)"
	celery -A src.api.celery_app flower --port=5555

setup-dev: install-dev ## Configuration complète pour le développement
	@echo "$(YELLOW)Configuration de l'environnement de développement...$(NC)"
	cp .env.example .env
	mkdir -p logs data/raw data/processed data/embeddings models
	@echo "$(GREEN)✅ Environnement de développement configuré!$(NC)"
	@echo "$(YELLOW)N'oubliez pas de modifier le fichier .env avec vos clés API$(NC)"

validate: lint test config-test ## Validation complète (linting + tests + config)
	@echo "$(GREEN)✅ Validation complète terminée!$(NC)"

benchmark: ## Lance les benchmarks de performance
	@echo "$(YELLOW)Lancement des benchmarks...$(NC)"
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only
	@echo "$(GREEN)✅ Benchmarks terminés!$(NC)"

docs: ## Génère la documentation
	@echo "$(YELLOW)Génération de la documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✅ Documentation générée dans docs/_build/html/$(NC)"

security-check: ## Vérification de sécurité
	@echo "$(YELLOW)Vérification de sécurité...$(NC)"
	bandit -r src/ -f json -o security-report.json
	safety check --json --output security-deps.json
	@echo "$(GREEN)✅ Vérification de sécurité terminée!$(NC)"

migrate: ## Lance les migrations de base de données
	@echo "$(YELLOW)Lancement des migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✅ Migrations terminées!$(NC)"

seed-data: ## Charge des données d'exemple
	@echo "$(YELLOW)Chargement des données d'exemple...$(NC)"
	$(PYTHON) scripts/seed_database.py
	@echo "$(GREEN)✅ Données d'exemple chargées!$(NC)"

backup: ## Sauvegarde de la base de données
	@echo "$(YELLOW)Sauvegarde de la base de données...$(NC)"
	$(PYTHON) scripts/backup_database.py
	@echo "$(GREEN)✅ Sauvegarde terminée!$(NC)"

status: ## Affiche le statut du système
	@echo "$(GREEN)Neural Chat Engine - System Status$(NC)"
	@echo "=================================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Docker version: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Git status: $$(git status --porcelain | wc -l) files modified"
	@echo "Environment: $$(cat .env 2>/dev/null | grep ENVIRONMENT || echo 'No .env file')"

# Règle par défaut
.DEFAULT_GOAL := help