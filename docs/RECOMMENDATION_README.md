# Système de Recommandations Contextuelles

Le système de recommandations contextuelles fournit des suggestions proactives et personnalisées basées sur le contexte de conversation, l'historique utilisateur et les patterns comportementaux.

## Vue d'ensemble

Le système implémente une approche hybride combinant :

- **Filtrage Collaboratif** : Recommandations basées sur les comportements d'utilisateurs similaires
- **Recommandations Basées sur le Contenu** : Suggestions basées sur les caractéristiques des éléments
- **Approche Hybride** : Combinaison optimisée des deux méthodes
- **Recommandations Contextuelles** : Adaptation basée sur le contexte actuel

## Architecture

```
src/recommendations/
├── __init__.py                    # Exports du module
├── recommendation_engine.py       # Moteur principal
├── recommendation_types.py        # Types de données
└── tests/
    └── test_recommendation_engine.py
```

## Composants Principaux

### RecommendationEngine

Le moteur principal gère l'ensemble du processus de recommandation :

```python
from src.recommendations import RecommendationEngine, UserProfile, ItemCatalog

# Initialisation
engine = RecommendationEngine(user_profiles, item_catalog)

# Génération de recommandations
result = engine.generate_proactive_suggestions(user_profile, context)
```

### Types de Données

#### UserProfile
Profil utilisateur avec historique et préférences :
```python
profile = UserProfile(
    user_id="user123",
    demographics={"age": 25, "location": "Paris"},
    preferences={"item1": 0.8, "item2": 0.6}
)
```

#### Item
Élément du catalogue avec métadonnées :
```python
item = Item(
    item_id="item1",
    title="MacBook Pro",
    category="technology",
    tags={"ordinateur", "apple", "professionnel"}
)
```

#### ContextSnapshot
Capture du contexte utilisateur :
```python
context = ContextSnapshot(
    context_type=ContextType.CONVERSATION,
    conversation_context="Je cherche un restaurant italien",
    time_of_day="19:30"
)
```

## Algorithmes de Recommandation

### 1. Filtrage Collaboratif

Identifie les utilisateurs similaires et recommande des éléments appréciés par ces utilisateurs :

```python
collaborative_score = engine._calculate_collaborative_score(user_profile, item)
```

**Avantages :**
- Découvre de nouveaux centres d'intérêt
- Basé sur le comportement réel des utilisateurs

**Limites :**
- Problème du démarrage à froid
- Sparsité des données

### 2. Recommandations Basées sur le Contenu

Analyse la similarité entre les éléments appréciés et les candidats :

```python
content_score = engine._calculate_content_score(user_profile, item)
```

**Avantages :**
- Fonctionne même avec peu de données
- Explications intuitives

**Limites :**
- Pas de découverte de nouveaux domaines
- Dépend de la qualité des métadonnées

### 3. Approche Hybride

Combine les scores collaboratifs et basés sur le contenu :

```python
hybrid_score = (
    collaborative_score * collaborative_weight +
    content_score * content_weight +
    context_score * context_weight
)
```

### 4. Recommandations Contextuelles

Adapte les suggestions selon le contexte :

- **Conversation** : Analyse des topics et sentiments
- **Temps** : Adaptation horaire (matinée, soirée, etc.)
- **Situation** : Événements spéciaux, urgences

## Utilisation

### Configuration

```python
config = {
    'max_recommendations': 20,
    'min_similarity_threshold': 0.1,
    'context_weight': 0.3,
    'collaborative_weight': 0.4,
    'content_weight': 0.3,
    'temporal_decay_factor': 0.95
}
```

### Génération de Recommandations

```python
# Analyse du contexte
context = engine.analyze_conversation_context("Je veux manger italien ce soir")

# Génération de suggestions
result = engine.generate_proactive_suggestions(
    user_profile=user_profile,
    context=context,
    max_suggestions=5
)

# Affichage des résultats
for rec in result.recommendations:
    item = catalog.get_item(rec.item_id)
    print(f"{item.title}: {rec.score:.2f} - {rec.explanation}")
```

### Apprentissage des Préférences

```python
# Nouvelles interactions
new_interactions = [
    UserInteraction(user_id, item_id, InteractionType.LIKE)
]

# Apprentissage
engine.learn_user_preferences(new_interactions)

# Feedback utilisateur
feedback = [
    FeedbackData(user_id, item_id, "accepted", 1.0)
]
engine._learn_from_feedback(feedback)
```

## Analyse Contextuelle

### Analyse de Conversation

Le système analyse automatiquement le texte de conversation :

```python
features = engine._extract_conversation_features(conversation)
# Retourne: topics, sentiment, urgency, length, etc.
```

### Détection de Topics

Topics automatiquement détectés :
- `technology` : ordinateur, logiciel, app
- `food` : restaurant, cuisine, recette
- `travel` : voyage, hôtel, destination

### Analyse Temporelle

Adaptation basée sur :
- Heure de la journée
- Jour de la semaine
- Saison
- Urgence détectée

## Métriques de Performance

### Suivi Automatique

```python
stats = engine.get_performance_stats()
print(f"Recommandations: {stats['total_recommendations']}")
print(f"Temps moyen: {stats['average_processing_time']:.3f}s")
```

### Métriques Clés

- **Précision** : Taux de recommandations acceptées
- **Rappel** : Couverture des éléments pertinents
- **Diversité** : Équilibre des catégories
- **Nouveauté** : Découverte de nouveaux éléments
- **Temps de réponse** : Performance en temps réel

## Gestion du Cache

### Optimisations

- **Cache de similarité** : Évite les recalculs coûteux
- **Vecteurs utilisateur** : Matrices pré-calculées
- **Nettoyage automatique** : Gestion de la mémoire

### Configuration

```python
config = {
    'cache_max_size': 10000,
    'model_update_interval_hours': 24
}
```

## Persistance des Modèles

### Sauvegarde

```python
# Sauvegarde des modèles
engine.save_models("models/recommendations/")

# Sauvegarde de la configuration
engine.save_config("config/recommendations.json")
```

### Chargement

```python
# Chargement des modèles
engine.load_models("models/recommendations/")
```

## Explications des Recommandations

### Génération Automatique

```python
explanations = engine.explain_recommendations(recommendations, context)

for item_id, explanation in explanations.items():
    print(f"{item_id}: {explanation}")
```

### Types d'Explications

- **Collaboratif** : "Recommandé car des utilisateurs similaires l'ont apprécié"
- **Contenu** : "Similaire à ce que vous avez aimé auparavant"
- **Contextuel** : "Pertinent pour votre conversation actuelle"
- **Hybride** : Combinaison des explications

## Tests et Validation

### Suite de Tests

```bash
# Exécution des tests
python -m pytest src/recommendations/tests/ -v

# Tests de performance
python -m pytest src/recommendations/tests/ -k "performance"
```

### Métriques de Test

- **Couverture** : > 90% du code testé
- **Performance** : < 100ms par recommandation
- **Précision** : > 80% de recommandations pertinentes

## Exemple Complet

Voir `examples/recommendation_demo.py` pour un exemple complet incluant :

- Configuration du système
- Création de données d'exemple
- Génération de recommandations
- Apprentissage et feedback
- Monitoring de performance

## Optimisations Futures

### Améliorations Planifiées

1. **Deep Learning** : Modèles neuronaux pour l'embedding
2. **Streaming** : Recommandations en temps réel
3. **Multi-modal** : Intégration vision/audio
4. **Federated Learning** : Apprentissage distribué
5. **A/B Testing** : Tests automatisés des algorithmes

### Scalabilité

- **Base de données** : Support PostgreSQL/MongoDB
- **Cache distribué** : Redis pour le cache
- **API REST** : Service de recommandations
- **Microservices** : Architecture distribuée

## Dépannage

### Problèmes Courants

1. **Cache plein** : Augmenter `cache_max_size`
2. **Performance lente** : Activer GPU ou réduire `max_candidates`
3. **Recommandations pauvres** : Vérifier la qualité des données
4. **Mémoire insuffisante** : Réduire la taille des matrices

### Logs et Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Logs détaillés
engine.logger.setLevel(logging.DEBUG)
```

## Contribution

### Développement

1. Ajouter des tests pour toute nouvelle fonctionnalité
2. Documenter les paramètres et méthodes
3. Respecter les patterns de performance
4. Optimiser pour la scalabilité

### Code Quality

- **Linting** : flake8, black
- **Tests** : pytest avec couverture > 90%
- **Documentation** : docstrings complètes
- **Performance** : benchmarks automatisés