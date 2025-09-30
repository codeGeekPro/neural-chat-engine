# Neural Chat Engine Core

Ce dossier contient le cœur du moteur de chat neuronal, responsable du traitement des conversations et de la génération des réponses.

## Structure

- `engine.py` : Classe principale du moteur de chat
- `model_manager.py` : Gestionnaire des modèles de langage
- `conversation.py` : Gestion des conversations et du contexte
- `context.py` : Gestion du contexte et de la mémoire
- `events.py` : Système d'événements du moteur
- `response.py` : Génération et formatage des réponses

## Fonctionnalités

- Gestion intelligente du contexte de conversation
- Support de multiples modèles de langage
- Système d'événements pour les hooks personnalisés
- Gestion de la mémoire et du contexte long-terme
- Génération de réponses optimisée

## Utilisation

Exemple d'utilisation basique du moteur :

```python
from engine import ChatEngine

engine = ChatEngine()
response = await engine.process_message("Bonjour !")
print(response.text)
```

## Extensibilité

Le moteur est conçu pour être facilement extensible via :

- Hooks d'événements personnalisés
- Plugins de traitement
- Intégration de nouveaux modèles
- Stratégies de gestion du contexte personnalisées