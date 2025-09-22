# ====================
# Production Dockerfile - Neural Chat Engine
# ====================

FROM python:3.11-slim as base

# Métadonnées
LABEL maintainer="CodeGeekPro <contact@codegeekpro.com>"
LABEL description="Neural Chat Engine - Chatbot IA Avancé"
LABEL version="0.1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1

# ====================
# Étape de construction
# ====================
FROM base as builder

# Installer les dépendances système pour la compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances
COPY requirements.txt setup.py ./
COPY src/ ./src/

# Installer les dépendances Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -e .

# ====================
# Étape de production
# ====================
FROM base as production

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root
RUN groupadd -r neural && useradd -r -g neural neural

# Créer les répertoires de travail
RUN mkdir -p /app /app/data /app/logs /app/models
WORKDIR /app

# Copier l'environnement Python depuis le builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copier le code source
COPY --chown=neural:neural . .

# Changer vers l'utilisateur non-root
USER neural

# Exposer les ports
EXPOSE 8000 8501

# Variables d'environnement par défaut
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande par défaut
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ====================
# Dockerfile pour développement
# ====================
FROM python:3.11-slim as development

# Variables d'environnement pour développement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=development

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt setup.py ./

# Installer les dépendances avec les outils de développement
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyter ipython pytest black isort flake8 mypy

# Exposer les ports pour développement
EXPOSE 8000 8501 8888

# Commande par défaut pour développement
CMD ["bash"]