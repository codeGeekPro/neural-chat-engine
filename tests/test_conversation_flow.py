import pytest
import asyncio
from src.core.chatbot_engine import ChatbotEngine

class TestConversationFlow:
    def setup_method(self):
        """Initialize test environment before each test"""
        self.engine = ChatbotEngine()
        self.test_user_id = "test_user_123"

    def test_intent_classification_accuracy(self):
        """Test accuracy of intent classification system"""
        test_cases = [
            ("Je voudrais réserver un billet d'avion", "reservation"),
            ("Quel temps fera-t-il demain ?", "meteo"),
            ("Au revoir", "fin_conversation"),
            ("Merci beaucoup", "remerciement")
        ]
        
        for input_text, expected_intent in test_cases:
            intent = self.engine.classify_intent(input_text)
            assert intent == expected_intent, f"Intent mismatch for '{input_text}'"

    def test_multi_turn_conversation(self):
        """Test handling of multi-turn conversations"""
        conversation = [
            ("Bonjour", "salutation"),
            ("Je cherche un restaurant", "recherche_restaurant"),
            ("Italien, si possible", "specification_cuisine"),
            ("Pour ce soir", "specification_temps")
        ]
        
        context = {}
        for user_input, expected_intent in conversation:
            response = self.engine.process_message(user_input, context)
            assert response is not None
            assert "intent" in response
            assert response["intent"] == expected_intent

    def test_context_preservation(self):
        """Test context preservation across conversation turns"""
        context = {}
        
        # Premier tour : établir le contexte
        response1 = self.engine.process_message(
            "Je cherche un restaurant italien", 
            context
        )
        assert "cuisine_type" in context
        assert context["cuisine_type"] == "italien"
        
        # Deuxième tour : utiliser le contexte
        response2 = self.engine.process_message(
            "Pour ce soir", 
            context
        )
        assert "timing" in context
        assert context["timing"] == "ce_soir"
        
        # Vérifier que le contexte précédent est préservé
        assert context["cuisine_type"] == "italien"

    def test_response_quality_metrics(self):
        """Test quality metrics of generated responses"""
        test_inputs = [
            "Bonjour, comment ça va ?",
            "Pouvez-vous m'aider avec ma réservation ?",
            "Je ne comprends pas"
        ]
        
        for input_text in test_inputs:
            response = self.engine.generate_response(input_text)
            
            # Vérifier la cohérence
            assert len(response) > 0
            
            # Vérifier le format
            assert isinstance(response, str)
            
            # Vérifier les métriques de qualité
            quality_metrics = self.engine.evaluate_response_quality(
                input_text, 
                response
            )
            assert quality_metrics["coherence"] >= 0.7
            assert quality_metrics["relevance"] >= 0.7
            assert quality_metrics["fluency"] >= 0.8

    async def test_concurrent_conversations(self):
        """Test handling of multiple concurrent conversations"""
        async def simulate_conversation(user_id, messages):
            context = {}
            responses = []
            for message in messages:
                response = await self.engine.process_message_async(
                    message, 
                    context, 
                    user_id
                )
                responses.append(response)
            return responses

        # Simuler plusieurs conversations simultanées
        conversations = {
            "user1": ["Bonjour", "Je cherche un restaurant", "Italien"],
            "user2": ["Hi", "What's the weather?", "Thank you"],
            "user3": ["Hola", "Necesito ayuda", "Gracias"]
        }

        tasks = [
            simulate_conversation(user_id, messages)
            for user_id, messages in conversations.items()
        ]

        # Exécuter les conversations en parallèle
        results = await asyncio.gather(*tasks)

        # Vérifier que chaque conversation a reçu le bon nombre de réponses
        for user_id, responses in zip(conversations.keys(), results):
            assert len(responses) == len(conversations[user_id])
            
        # Vérifier qu'il n'y a pas eu de mélange de contextes
        for responses in results:
            assert all(r is not None for r in responses)