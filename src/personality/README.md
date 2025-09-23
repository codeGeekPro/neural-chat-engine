# Moteur de Personnalité Adaptative

Le moteur de personnalité adaptative permet au chatbot d'analyser le style de communication de l'utilisateur et d'adapter ses réponses en conséquence.

## Fonctionnalités

### Analyse du Style de Communication
- Détection automatique du style (formel, casual, technique, simple, etc.)
- Analyse linguistique et émotionnelle des messages
- Calcul de métriques de complexité et formalité

### Profils de Personnalité
- Création de profils utilisateurs persistants
- Suivi des dimensions de personnalité :
  - **Formalité** : Formal vs Casual
  - **Complexité** : Technique vs Simple
  - **Amicalité** : Amical vs Professionnel
  - **Créativité** : Créatif vs Direct

### Adaptation des Réponses
- Modification automatique du ton des réponses
- Adaptation progressive pour maintenir la cohérence
- Apprentissage des préférences utilisateur

### Cohérence et Apprentissage
- Maintenance de la cohérence dans les conversations
- Apprentissage continu des préférences
- Système de feedback pour améliorer les adaptations

## Utilisation

```python
from src.personality.personality_engine import PersonalityEngine

# Initialisation
engine = PersonalityEngine()

# Analyse d'une conversation
conversation = [
    {"role": "user", "content": "Salut ! Peux-tu m'aider ?"},
    {"role": "assistant", "content": "Bien sûr !"}
]

style_analysis = engine.analyze_user_communication_style(conversation)
print(f"Style détecté: {style_analysis.detected_style.value}")

# Création d'un profil
profile = engine.create_personality_profile(conversation, "user_id")

# Adaptation d'une réponse
base_response = "Voici une explication technique..."
adaptation = engine.adapt_response_tone(base_response, profile)
adapted_response = adaptation.adapted_tone
```

## Architecture

```
src/personality/
├── __init__.py                 # Exports du module
├── personality_engine.py       # Moteur principal
└── personality_types.py        # Types et structures de données

tests/personality/
└── test_personality_engine.py  # Tests unitaires

examples/
└── personality_demo.py         # Démonstration
```

## Métriques Analysées

### Caractéristiques Linguistiques
- Longueur moyenne des phrases
- Richesse du vocabulaire
- Ratio de questions/exclamations
- Score de créativité

### Indicateurs Émotionnels
- Niveau d'amicalité
- Degré d'urgence
- Niveau de politesse
- Enthousiasme

### Métriques de Complexité
- Ratio de termes techniques
- Complexité des structures de phrases
- Score global de complexité

## Styles de Communication Supportés

- **FORMAL** : Langage soutenu, poli
- **CASUAL** : Langage familier, détendu
- **TECHNICAL** : Vocabulaire spécialisé
- **SIMPLE** : Explications accessibles
- **FRIENDLY** : Ton amical et engageant
- **PROFESSIONAL** : Approche sérieuse
- **CREATIVE** : Style original et imaginatif
- **DIRECT** : Communication directe et concise

## Personnalisation

Le moteur peut être personnalisé via :
- Modèles d'embedding différents
- Seuils de détection de style
- Règles d'adaptation spécifiques
- Métriques personnalisées

## Tests

```bash
# Exécuter les tests
pytest tests/personality/

# Avec couverture
pytest --cov=src/personality tests/personality/
```

## Démonstration

Voir `examples/personality_demo.py` pour un exemple complet d'utilisation.