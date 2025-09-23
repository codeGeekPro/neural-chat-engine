"""Exemple d'utilisation du système d'apprentissage continu."""

import torch
import torch.nn as nn
from datetime import datetime

from src.learning.continual_learning import ContinualLearningSystem
from src.learning.learning_types import LearningTask


def create_sample_models():
    """Crée des modèles d'exemple pour la démonstration."""
    # Modèle simple de classification
    conversation_model = nn.Sequential(
        nn.Linear(768, 256),  # Embeddings d'entrée
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)  # 10 classes de conversation
    )

    # Modèle d'analyse d'émotion
    emotion_model = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 7)  # 7 émotions
    )

    return {
        "conversation_processing": conversation_model,
        "emotion_analysis": emotion_model
    }


def demonstrate_continual_learning():
    """Démontre l'utilisation du système d'apprentissage continu."""

    print("=== Initialisation du système d'apprentissage continu ===")

    # Crée les modèles
    models = create_sample_models()

    # Initialise le système
    cl_system = ContinualLearningSystem(
        base_models=models,
        learning_rate=1e-4,
        device="cpu"
    )

    print(f"Système initialisé avec {len(models)} modèles")

    # Données d'exemple
    new_conversations = [
        {
            "user_input": "Bonjour, pouvez-vous m'aider avec Python ?",
            "bot_response": "Bien sûr ! Que voulez-vous savoir sur Python ?",
            "emotion": "curious",
            "topic": "programming",
            "timestamp": datetime.now().isoformat()
        },
        {
            "user_input": "J'ai une erreur dans mon code.",
            "bot_response": "Pouvez-vous partager l'erreur que vous obtenez ?",
            "emotion": "frustrated",
            "topic": "debugging",
            "timestamp": datetime.now().isoformat()
        },
        {
            "user_input": "Merci beaucoup pour votre aide !",
            "bot_response": "De rien, n'hésitez pas si vous avez d'autres questions.",
            "emotion": "grateful",
            "topic": "general",
            "timestamp": datetime.now().isoformat()
        }
    ]

    print("\n=== Mise à jour incrémentale ===")

    # Met à jour le modèle de traitement de conversation
    result = cl_system.update_model_incrementally(
        new_conversations,
        "conversation_processing",
        batch_size=8,
        epochs=2
    )

    print(f"Mise à jour réussie: {result['success']}")
    print(f"Nouvelle version: {result['version']}")
    print(".3f")

    print("\n=== Prévention de l'oubli catastrophique ===")

    # Données des anciennes tâches pour EWC
    old_tasks_data = {
        "conversation_processing": [
            {
                "user_input": "Comment ça va ?",
                "bot_response": "Très bien, merci !",
                "emotion": "casual",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

    # Applique EWC
    ewc_success = cl_system.prevent_catastrophic_forgetting(
        old_tasks_data, "conversation_processing"
    )

    print(f"EWC appliqué avec succès: {ewc_success}")

    print("\n=== Évaluation de la dérive ===")

    # Évalue la performance
    test_data = [{"conversations": new_conversations}]
    performance = cl_system.evaluate_model_drift(
        test_data, "conversation_processing"
    )

    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")

    print("\n=== Historique des versions ===")

    # Affiche l'historique
    history = cl_system.get_model_history("conversation_processing")
    print(f"Nombre de versions: {len(history)}")

    for version in history[-3:]:  # Affiche les 3 dernières
        print(f"  Version {version['version']}: {version['status']} "
              ".3f")

    print("\n=== Planification du réentraînement ===")

    # Vérifie quels modèles nécessitent un réentraînement
    models_to_retrain = cl_system.schedule_retraining(performance_threshold=0.85)

    print("Modèles nécessitant un réentraînement:")
    for model_name, reasons in models_to_retrain.items():
        print(f"  {model_name}:")
        for reason in reasons:
            print(f"    - {reason}")

    print("\n=== Tendances de performance ===")

    # Analyse les tendances
    trends = cl_system.get_performance_trends("conversation_processing", days=30)

    if trends:
        print("Évolution de la performance:")
        print(".3f")
        print(".3f")
        print(".3f")
    else:
        print("Pas assez de données pour analyser les tendances")

    print("\n=== Rollback de version ===")

    # Simule un rollback
    if len(history) > 1:
        rollback_success = cl_system.rollback_to_previous_version(
            "conversation_processing"
        )
        print(f"Rollback réussi: {rollback_success}")
        print(f"Version actuelle: {cl_system.current_versions['conversation_processing']}")
    else:
        print("Pas de versions précédentes pour le rollback")

    print("\n=== Résumé ===")
    print("Le système d'apprentissage continu permet de:")
    print("• Mettre à jour les modèles de manière incrémentale")
    print("• Prévenir l'oubli catastrophique avec EWC")
    print("• Surveiller la dérive de performance")
    print("• Gérer les versions et permettre les rollbacks")
    print("• Planifier automatiquement le réentraînement")


if __name__ == "__main__":
    demonstrate_continual_learning()