# ====================
# Production Dockerfile - Neural Chat Engine
# ====================

FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu24.04 AS base

# Métadonnées
LABEL maintainer="CodeGeekPro <contact@codegeekpro.com>"
LABEL description="Neural Chat Engine - Chatbot IA Avancé"
LABEL version="0.1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/models/cache

# ====================
# Étape de construction
# ====================
FROM base as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/requirements.txt ./
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

RUN mkdir -p /app /app/data /app/logs /app/models $MODEL_CACHE_DIR
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=neural:neural . .
RUN chmod 700 $MODEL_CACHE_DIR

USER neural

EXPOSE 8000 8501

# Variables d'environnement par défaut
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


# Signal handling pour une sortie propre
STOPSIGNAL SIGTERM

# Commande par défaut
CMD ["python3.11", "scripts/start.py"]

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