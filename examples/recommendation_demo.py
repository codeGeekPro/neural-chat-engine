"""Exemple d'utilisation du système de recommandations."""

import asyncio
from datetime import datetime
from src.recommendations import (
    RecommendationEngine,
    UserProfile,
    Item,
    ItemCatalog,
    UserInteraction,
    ContextSnapshot,
    InteractionType,
    ContextType,
    FeedbackData
)


def create_sample_data():
    """Crée des données d'exemple pour la démonstration."""
    # Création du catalogue d'éléments
    catalog = ItemCatalog()

    # Éléments technologiques
    tech_items = [
        Item(
            item_id="laptop_pro",
            title="MacBook Pro M3",
            description="Ordinateur portable professionnel avec puce M3",
            category="technology",
            tags={"ordinateur", "macbook", "professionnel", "m3"},
            popularity_score=0.9
        ),
        Item(
            item_id="iphone_15",
            title="iPhone 15 Pro",
            description="Smartphone dernière génération avec appareil photo avancé",
            category="technology",
            tags={"smartphone", "iphone", "photo", "5g"},
            popularity_score=0.8
        ),
        Item(
            item_id="airpods",
            title="AirPods Pro",
            description="Écouteurs sans fil avec réduction de bruit",
            category="technology",
            tags={"écouteurs", "bluetooth", "audio", "apple"},
            popularity_score=0.7
        )
    ]

    # Éléments culinaires
    food_items = [
        Item(
            item_id="pasta_carbonara",
            title="Recette Carbonara Authentique",
            description="Recette traditionnelle de pâtes carbonara italiennes",
            category="food",
            tags={"pasta", "italien", "recette", "traditionnel"},
            popularity_score=0.8
        ),
        Item(
            item_id="sushi_masterclass",
            title="Cours de Sushi",
            description="Apprenez à préparer des sushis avec un chef expérimenté",
            category="food",
            tags={"sushi", "japonais", "cuisine", "atelier"},
            popularity_score=0.6
        ),
        Item(
            item_id="vin_rouge_bordeaux",
            title="Bordeaux Rouge 2018",
            description="Vin rouge de Bordeaux millésime 2018",
            category="food",
            tags={"vin", "bordeaux", "rouge", "millésime"},
            popularity_score=0.7
        )
    ]

    # Éléments de voyage
    travel_items = [
        Item(
            item_id="paris_weekend",
            title="Week-end à Paris",
            description="Séjour romantique de 3 jours à Paris avec hôtel 4*",
            category="travel",
            tags={"paris", "weekend", "romantique", "hôtel"},
            popularity_score=0.8
        ),
        Item(
            item_id="ski_alps",
            title="Séjour Ski Alpes",
            description="Une semaine de ski dans les Alpes françaises",
            category="travel",
            tags={"ski", "alpes", "neige", "montagne"},
            popularity_score=0.6
        )
    ]

    # Ajouter tous les éléments au catalogue
    for item in tech_items + food_items + travel_items:
        catalog.add_item(item)

    # Création des profils utilisateurs
    user_profiles = {}

    # Utilisateur tech enthusiast
    user1 = UserProfile(
        user_id="tech_lover",
        demographics={"age": 28, "gender": "M", "profession": "developer", "location": "Paris"},
        preferences={
            "laptop_pro": 0.9,
            "iphone_15": 0.8,
            "airpods": 0.7
        }
    )

    # Interactions passées
    interactions1 = [
        UserInteraction("tech_lover", "laptop_pro", InteractionType.PURCHASE, datetime.now().replace(hour=14)),
        UserInteraction("tech_lover", "iphone_15", InteractionType.LIKE, datetime.now().replace(hour=10)),
        UserInteraction("tech_lover", "airpods", InteractionType.VIEW, datetime.now().replace(hour=16)),
    ]
    for interaction in interactions1:
        user1.add_interaction(interaction)

    user_profiles["tech_lover"] = user1

    # Utilisateur food lover
    user2 = UserProfile(
        user_id="food_enthusiast",
        demographics={"age": 35, "gender": "F", "profession": "chef", "location": "Lyon"},
        preferences={
            "pasta_carbonara": 0.9,
            "sushi_masterclass": 0.8,
            "vin_rouge_bordeaux": 0.7
        }
    )

    interactions2 = [
        UserInteraction("food_enthusiast", "pasta_carbonara", InteractionType.PURCHASE, datetime.now().replace(hour=12)),
        UserInteraction("food_enthusiast", "sushi_masterclass", InteractionType.LIKE, datetime.now().replace(hour=15)),
        UserInteraction("food_enthusiast", "vin_rouge_bordeaux", InteractionType.SHARE, datetime.now().replace(hour=20)),
    ]
    for interaction in interactions2:
        user2.add_interaction(interaction)

    user_profiles["food_enthusiast"] = user2

    # Utilisateur travel lover
    user3 = UserProfile(
        user_id="travel_addict",
        demographics={"age": 42, "gender": "M", "profession": "sales", "location": "Marseille"},
        preferences={
            "paris_weekend": 0.8,
            "ski_alps": 0.6
        }
    )

    interactions3 = [
        UserInteraction("travel_addict", "paris_weekend", InteractionType.LIKE, datetime.now().replace(hour=9)),
        UserInteraction("travel_addict", "ski_alps", InteractionType.VIEW, datetime.now().replace(hour=11)),
    ]
    for interaction in interactions3:
        user3.add_interaction(interaction)

    user_profiles["travel_addict"] = user3

    return user_profiles, catalog


def demonstrate_basic_recommendations():
    """Démontre les recommandations de base."""
    print("🎯 Démonstration des Recommandations de Base")
    print("=" * 50)

    user_profiles, catalog = create_sample_data()
    engine = RecommendationEngine(user_profiles, catalog)

    # Recommandations pour l'utilisateur tech
    user_profile = user_profiles["tech_lover"]
    print(f"\n📱 Recommandations pour {user_profile.user_id} (Technologie):")

    result = engine.generate_proactive_suggestions(user_profile, max_suggestions=3)

    for i, rec in enumerate(result.recommendations, 1):
        item = catalog.get_item(rec.item_id)
        print(f"{i}. {item.title} (Score: {rec.score:.2f}, Confiance: {rec.confidence:.2f})")
        print(f"   📝 {rec.explanation}")

    print(f"\n⏱️ Temps de traitement: {result.processing_time:.3f}s")
    print(f"📊 Candidats évalués: {result.total_candidates}")


def demonstrate_contextual_recommendations():
    """Démontre les recommandations contextuelles."""
    print("\n🎭 Démonstration des Recommandations Contextuelles")
    print("=" * 50)

    user_profiles, catalog = create_sample_data()
    engine = RecommendationEngine(user_profiles, catalog)

    # Contexte de conversation food
    conversation_context = ContextSnapshot(
        context_type=ContextType.CONVERSATION,
        conversation_context="J'ai envie de cuisiner des pâtes italiennes ce soir avec un bon vin rouge.",
        time_of_day="19:30",
        day_of_week="Friday"
    )

    print("
💬 Contexte: 'J'ai envie de cuisiner des pâtes italiennes ce soir avec un bon vin rouge.'")
    print(f"🕐 Heure: {conversation_context.time_of_day}, Jour: {conversation_context.day_of_week}")

    # Analyser le contexte
    analyzed_context = engine.analyze_conversation_context(conversation_context.conversation_context)
    print("
🔍 Analyse du contexte:"    print(f"   - Topics détectés: {analyzed_context.features.get('topics', [])}")
    print(".2f"    print(".2f"    print(".2f"
    # Générer des recommandations contextuelles pour l'utilisateur food
    user_profile = user_profiles["food_enthusiast"]
    result = engine.generate_proactive_suggestions(user_profile, analyzed_context, max_suggestions=3)

    print(f"\n🍝 Recommandations contextuelles pour {user_profile.user_id}:")

    for i, rec in enumerate(result.recommendations, 1):
        item = catalog.get_item(rec.item_id)
        print(f"{i}. {item.title} (Score: {rec.score:.2f})")
        print(f"   📝 {rec.explanation}")


def demonstrate_learning_and_feedback():
    """Démontre l'apprentissage et le feedback."""
    print("\n🧠 Démonstration de l'Apprentissage et Feedback")
    print("=" * 50)

    user_profiles, catalog = create_sample_data()
    engine = RecommendationEngine(user_profiles, catalog)

    user_profile = user_profiles["tech_lover"]
    print(f"Utilisateur: {user_profile.user_id}")
    print(f"Préférences initiales: {user_profile.preferences}")

    # Simuler de nouvelles interactions
    new_interactions = [
        UserInteraction(
            user_id="tech_lover",
            item_id="airpods",
            interaction_type=InteractionType.PURCHASE,
            timestamp=datetime.now()
        ),
        UserInteraction(
            user_id="tech_lover",
            item_id="pasta_carbonara",
            interaction_type=InteractionType.DISLIKE,
            timestamp=datetime.now()
        )
    ]

    print("
🆕 Nouvelles interactions:"    for interaction in new_interactions:
        item = catalog.get_item(interaction.item_id)
        print(f"   - {interaction.interaction_type.value}: {item.title}")

    # Apprendre des nouvelles interactions
    engine.learn_user_preferences(new_interactions)

    print("
📈 Préférences après apprentissage:"    print(f"   {user_profile.preferences}")

    # Simuler du feedback
    feedback_data = [
        FeedbackData(
            user_id="tech_lover",
            item_id="airpods",
            feedback_type="accepted",
            feedback_score=1.0,
            recommended_at=datetime.now()
        )
    ]

    print("
💬 Feedback utilisateur:"    for feedback in feedback_data:
        item = catalog.get_item(feedback.item_id)
        print(f"   - {feedback.feedback_type}: {item.title} (Score: {feedback.feedback_score})")

    engine._learn_from_feedback(feedback_data)

    print("
🎯 Préférences finales après feedback:"    print(f"   {user_profile.preferences}")

    # Générer de nouvelles recommandations
    result = engine.generate_proactive_suggestions(user_profile, max_suggestions=2)
    print("
🔄 Nouvelles recommandations après apprentissage:"    for i, rec in enumerate(result.recommendations, 1):
        item = catalog.get_item(rec.item_id)
        print(f"{i}. {item.title} (Score: {rec.score:.2f})")


def demonstrate_performance_monitoring():
    """Démontre le monitoring de performance."""
    print("\n📊 Démonstration du Monitoring de Performance")
    print("=" * 50)

    user_profiles, catalog = create_sample_data()
    engine = RecommendationEngine(user_profiles, catalog)

    # Générer plusieurs recommandations pour mesurer la performance
    user_profile = user_profiles["tech_lover"]

    print("⏳ Génération de 5 séries de recommandations...")
    for i in range(5):
        result = engine.generate_proactive_suggestions(user_profile, max_suggestions=3)
        print(".3f"
    # Afficher les statistiques de performance
    stats = engine.get_performance_stats()

    print("
📈 Statistiques de performance:"    print(f"   - Recommandations totales: {stats['total_recommendations']}")
    print(".3f"    print(f"   - Taille du cache: {stats['cache_size']}")
    print(f"   - Utilisateurs totaux: {stats['total_users']}")
    print(f"   - Éléments totaux: {stats['total_items']}")

    print("
🤖 Statut des modèles:"    for model_name, model_info in stats['models_status'].items():
        expired = "❌ Expiré" if model_info['is_expired'] else "✅ Actif"
        print(f"   - {model_name}: {expired} (Dernier entraînement: {model_info['trained_at'][:19]})")


def demonstrate_explanations():
    """Démontre les explications de recommandations."""
    print("\n💡 Démonstration des Explications de Recommandations")
    print("=" * 50)

    user_profiles, catalog = create_sample_data()
    engine = RecommendationEngine(user_profiles, catalog)

    user_profile = user_profiles["food_enthusiast"]

    # Contexte de dîner
    context = ContextSnapshot(
        context_type=ContextType.TIME_BASED,
        time_of_day="20:00",
        day_of_week="Saturday"
    )

    result = engine.generate_proactive_suggestions(user_profile, context, max_suggestions=3)

    print("🍽️ Recommandations pour un samedi soir:")
    explanations = engine.explain_recommendations(result.recommendations, context)

    for rec in result.recommendations:
        item = catalog.get_item(rec.item_id)
        print(f"\n📦 {item.title}")
        print(f"   ⭐ Score: {rec.score:.2f}")
        print(f"   🎯 Type: {rec.recommendation_type.value}")
        print(f"   💬 Explication: {explanations[rec.item_id]}")

        if rec.similar_items:
            similar_names = []
            for similar_id in rec.similar_items:
                similar_item = catalog.get_item(similar_id)
                if similar_item:
                    similar_names.append(similar_item.title)
            print(f"   🔗 Similaire à: {', '.join(similar_names)}")


def main():
    """Fonction principale de démonstration."""
    print("🎯 Système de Recommandations Contextuelles")
    print("=" * 60)
    print("Démonstration des capacités avancées de recommandation")

    try:
        demonstrate_basic_recommendations()
        demonstrate_contextual_recommendations()
        demonstrate_learning_and_feedback()
        demonstrate_performance_monitoring()
        demonstrate_explanations()

        print("\n" + "=" * 60)
        print("✅ Démonstration terminée avec succès !")
        print("\n🚀 Capacités démontrées :")
        print("   • Recommandations basées sur les préférences utilisateur")
        print("   • Analyse contextuelle de conversation")
        print("   • Apprentissage continu des préférences")
        print("   • Feedback et ajustements en temps réel")
        print("   • Monitoring de performance")
        print("   • Explications transparentes des recommandations")

    except Exception as e:
        print(f"\n❌ Erreur lors de la démonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()