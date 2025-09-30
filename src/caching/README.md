# Système de Cache Intelligent

Ce module implémente un système de cache avancé avec support de la similarité sémantique et gestion intelligente de la mémoire pour les réponses et les embeddings.

## Caractéristiques Principales

### 1. Cache Sémantique
```python
cache = IntelligentCache(similarity_threshold=0.85)
result, similarity = cache.get("requête", semantic_search=True)
```

- Recherche par similarité sémantique
- Seuil de similarité configurable
- Utilisation de SentenceTransformers
- Cache des embeddings pour les performances

### 2. Gestion de la Mémoire
```python
cache = IntelligentCache(max_size_mb=1024)
```

Fonctionnalités :
- Limite de taille configurable
- Politique d'éviction intelligente
- Surveillance de l'utilisation mémoire
- Nettoyage automatique

### 3. Cache Distribué
```python
cache = IntelligentCache(
    distributed=True,
    redis_url="redis://localhost:6379"
)
```

- Support Redis intégré
- Synchronisation automatique
- Réplication des données
- Haute disponibilité

### 4. Gestion du TTL
```python
cache = IntelligentCache(ttl_hours=24)
```

- Expiration automatique des entrées
- TTL configurable par entrée
- Nettoyage périodique
- Optimisation de l'espace

## Configuration

### Installation

```bash
pip install sentence-transformers redis numpy
```

### Configuration de Base

```python
from caching.intelligent_cache import IntelligentCache

cache = IntelligentCache(
    max_size_mb=1024,
    similarity_threshold=0.85,
    ttl_hours=24,
    distributed=False
)
```

### Configuration Redis

```python
cache = IntelligentCache(
    distributed=True,
    redis_url="redis://username:password@host:6379",
    max_size_mb=2048
)
```

## Utilisation

### 1. Opérations Basiques

```python
# Ajout au cache
cache.set("clé", "valeur")

# Récupération simple
result, _ = cache.get("clé")

# Récupération sémantique
result, similarity = cache.get(
    "requête similaire",
    semantic_search=True
)
```

### 2. Gestion Avancée

```python
# Statistiques du cache
stats = cache.get_stats()
print(f"""
Statistiques du Cache:
- Entrées totales: {stats['total_entries']}
- Utilisation mémoire: {stats['memory_usage_percent']}%
- Taille moyenne: {stats['average_entry_size_kb']}KB
""")

# Nettoyage manuel
cache.clear()
```

## Fonctionnement Interne

### 1. Calcul de Similarité

Le système utilise :
- Embeddings via SentenceTransformer
- Similarité cosinus
- Mise en cache des embeddings

### 2. Politique d'Éviction

Critères de score :
- Fréquence d'accès
- Âge de l'entrée
- Taille en mémoire
- Dernier accès

### 3. Gestion Distribuée

Fonctionnalités Redis :
- Réplication automatique
- Persistence des données
- Gestion des timeouts
- Compression des données

## Optimisation des Performances

### 1. Paramètres Clés

```python
cache = IntelligentCache(
    max_size_mb=1024,          # Taille maximale
    similarity_threshold=0.85,  # Seuil de similarité
    ttl_hours=24,              # Durée de vie
)
```

### 2. Recommandations

- Ajuster `max_size_mb` selon la RAM disponible
- Régler `similarity_threshold` selon la précision souhaitée
- Optimiser `ttl_hours` selon l'actualité des données

## Monitoring et Maintenance

### 1. Métriques Disponibles

```python
stats = cache.get_stats()
```

Métriques clés :
- Taux de hit/miss
- Utilisation mémoire
- Âge des entrées
- Performance du cache

### 2. Maintenance

```python
# Nettoyage périodique
if stats['memory_usage_percent'] > 90:
    cache.clear()
```

## Dépannage

### Problèmes Courants

1. **Utilisation Mémoire Élevée**
   - Réduire `max_size_mb`
   - Diminuer `ttl_hours`
   - Augmenter la fréquence de nettoyage

2. **Cache Miss Fréquents**
   - Ajuster `similarity_threshold`
   - Vérifier la qualité des embeddings
   - Optimiser les clés de cache

3. **Latence Redis**
   - Vérifier la configuration réseau
   - Optimiser les timeouts
   - Considérer la réplication

## Exemples d'Utilisation Avancée

### 1. Cache Hybride

```python
cache = IntelligentCache(
    distributed=True,
    max_size_mb=512,  # Cache local
    redis_url="redis://localhost:6379"
)
```

### 2. Monitoring Temps Réel

```python
# Surveillance continue
import time

while True:
    stats = cache.get_stats()
    print(f"Utilisation mémoire: {stats['memory_usage_percent']}%")
    time.sleep(60)
```