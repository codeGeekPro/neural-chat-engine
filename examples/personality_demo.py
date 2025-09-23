"""Exemple d'utilisation du moteur de personnalité adaptative."""

from src.personality.personality_engine import PersonalityEngine
from src.personality.personality_types import PersonalityProfile, PersonalityDimension


def demonstrate_personality_engine():
    """Démontre l'utilisation du moteur de personnalité."""

    # Initialise le moteur
    engine = PersonalityEngine()

    # Exemple de conversation
    conversation = [
        {
            "role": "user",
            "content": "Salut ! Peux-tu m'expliquer simplement comment fonctionne un algorithme de graphe ?"
        },
        {
            "role": "assistant",
            "content": "Bien sûr ! Un algorithme de graphe travaille avec des nœuds connectés par des arêtes."
        },
        {
            "role": "user",
            "content": "Cool, mais c'est un peu technique. Tu peux donner un exemple concret ?"
        }
    ]

    print("=== Analyse du style de communication ===")
    style_analysis = engine.analyze_user_communication_style(conversation)
    print(f"Style détecté: {style_analysis.detected_style.value}")
    print(".2f")
    print(".2f")
    print(f"Score de formalité: {style_analysis.formality_score:.2f}")

    print("\n=== Création du profil de personnalité ===")
    profile = engine.create_personality_profile(conversation, "demo_user")
    print(f"Utilisateur: {profile.user_id}")
    print(f"Style de communication: {profile.communication_style.value}")
    print(f"Échantillons: {profile.sample_size}")
    print(f"Profil confiant: {profile.is_confident()}")

    print("\nDimensions de personnalité:")
    for dimension, value in profile.personality_dimensions.items():
        print(f"  {dimension.value}: {value:.2f}")

    print("\n=== Adaptation du ton des réponses ===")
    base_response = "Les algorithmes de graphe sont utilisés pour résoudre des problèmes complexes dans les réseaux."

    adaptation = engine.adapt_response_tone(base_response, profile)
    print(f"Réponse originale: {base_response}")
    print(f"Réponse adaptée: {adaptation.adapted_tone}")
    print(f"Confiance: {adaptation.confidence_score:.2f}")
    print(f"Raisonnement: {adaptation.reasoning}")

    print("\n=== Maintenance de cohérence ===")
    consistency = engine.maintain_personality_consistency(conversation, profile)
    print(f"Score de cohérence: {consistency['consistency_score']:.2f}")
    print(f"Recommandations: {consistency['recommendations']}")

    print("\n=== Apprentissage des préférences ===")
    feedback = [
        {"sentiment": "positive", "response_id": "resp1"},
        {"sentiment": "positive", "response_id": "resp2"}
    ]
    response_history = [
        {"id": "resp1", "content": "Salut ! Je t'explique ça simplement."},
        {"id": "resp2", "content": "Pas de problème, on va voir ça ensemble !"}
    ]

    preferences = engine.learn_user_preferences(feedback, response_history)
    print(f"Confiance d'apprentissage: {preferences['learning_confidence']:.2f}")
    print(f"Tons préférés: {preferences['preferred_tones']}")
    print(f"Patterns à éviter: {preferences['avoided_patterns']}")


if __name__ == "__main__":
    demonstrate_personality_engine()