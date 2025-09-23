"""Moteur de personnalité adaptative pour le chatbot."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from .personality_types import (
    CommunicationStyle,
    PersonalityDimension,
    PersonalityProfile,
    PersonalityAdaptation,
    StyleAnalysis
)


logger = logging.getLogger(__name__)


class PersonalityEngine:
    """Moteur de personnalité adaptative."""

    def __init__(
        self,
        personality_models_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialise le moteur de personnalité.

        Args:
            personality_models_path: Chemin vers les modèles de personnalité
            embedding_model: Modèle pour les embeddings
        """
        self.models_path = Path(personality_models_path) if personality_models_path else None
        self.embedding_model = SentenceTransformer(embedding_model)

        # Cache des profils
        self.user_profiles: Dict[str, PersonalityProfile] = {}

        # Modèles d'analyse de style
        self.style_analyzer = self._initialize_style_analyzer()

        # Normaliseur pour les caractéristiques
        self.scaler = StandardScaler()

        # Clusters de personnalités (pour l'apprentissage)
        self.personality_clusters = None

        logger.info("PersonalityEngine initialisé")

    def analyze_user_communication_style(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> StyleAnalysis:
        """Analyse le style de communication de l'utilisateur.

        Args:
            conversation_history: Historique des conversations

        Returns:
            Analyse du style de communication
        """
        if not conversation_history:
            return StyleAnalysis(
                detected_style=CommunicationStyle.CASUAL,
                style_confidence=0.0,
                linguistic_features={},
                emotional_indicators={},
                complexity_metrics={},
                formality_score=0.5
            )

        # Extrait les messages utilisateur
        user_messages = [
            msg["content"] for msg in conversation_history
            if msg.get("role") == "user"
        ]

        if not user_messages:
            return StyleAnalysis(
                detected_style=CommunicationStyle.CASUAL,
                style_confidence=0.0,
                linguistic_features={},
                emotional_indicators={},
                complexity_metrics={},
                formality_score=0.5
            )

        # Analyse linguistique
        linguistic_features = self._extract_linguistic_features(user_messages)
        emotional_indicators = self._extract_emotional_indicators(user_messages)
        complexity_metrics = self._extract_complexity_metrics(user_messages)

        # Calcule le score de formalité
        formality_score = self._calculate_formality_score(
            linguistic_features,
            emotional_indicators
        )

        # Détermine le style dominant
        detected_style, confidence = self._determine_communication_style(
            linguistic_features,
            emotional_indicators,
            complexity_metrics,
            formality_score
        )

        return StyleAnalysis(
            detected_style=detected_style,
            style_confidence=confidence,
            linguistic_features=linguistic_features,
            emotional_indicators=emotional_indicators,
            complexity_metrics=complexity_metrics,
            formality_score=formality_score
        )

    def create_personality_profile(
        self,
        user_interactions: List[Dict[str, Any]],
        user_id: str
    ) -> PersonalityProfile:
        """Crée un profil de personnalité pour l'utilisateur.

        Args:
            user_interactions: Interactions de l'utilisateur
            user_id: ID de l'utilisateur

        Returns:
            Profil de personnalité
        """
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
        else:
            profile = PersonalityProfile(user_id=user_id)
            self.user_profiles[user_id] = profile

        # Analyse le style de communication
        style_analysis = self.analyze_user_communication_style(user_interactions)

        # Met à jour les dimensions de personnalité
        profile.personality_dimensions[PersonalityDimension.FORMALITY] = style_analysis.formality_score
        profile.personality_dimensions[PersonalityDimension.COMPLEXITY] = style_analysis.complexity_metrics.get("avg_complexity", 0.5)
        profile.personality_dimensions[PersonalityDimension.FRIENDLINESS] = style_analysis.emotional_indicators.get("friendliness", 0.5)
        profile.personality_dimensions[PersonalityDimension.CREATIVITY] = style_analysis.linguistic_features.get("creativity_score", 0.5)

        # Met à jour le style de communication
        profile.communication_style = style_analysis.detected_style

        # Met à jour l'historique
        profile.update_from_interaction({
            "style_analysis": style_analysis.to_dict(),
            "interaction_count": len(user_interactions)
        }, confidence=style_analysis.style_confidence)

        return profile

    def adapt_response_tone(
        self,
        base_response: str,
        target_personality: PersonalityProfile,
        context: Optional[Dict[str, Any]] = None
    ) -> PersonalityAdaptation:
        """Adapte le ton de la réponse selon la personnalité cible.

        Args:
            base_response: Réponse de base
            target_personality: Personnalité cible
            context: Contexte additionnel

        Returns:
            Adaptation de personnalité
        """
        original_tone = self._analyze_response_tone(base_response)
        adaptation_factors = {}

        # Adapte selon chaque dimension
        adapted_response = base_response

        # Formalité
        formality_factor = target_personality.personality_dimensions[PersonalityDimension.FORMALITY]
        adapted_response, formality_change = self._adapt_formality(
            adapted_response, formality_factor
        )
        adaptation_factors["formality"] = formality_change

        # Complexité
        complexity_factor = target_personality.personality_dimensions[PersonalityDimension.COMPLEXITY]
        adapted_response, complexity_change = self._adapt_complexity(
            adapted_response, complexity_factor
        )
        adaptation_factors["complexity"] = complexity_change

        # Amicalité
        friendliness_factor = target_personality.personality_dimensions[PersonalityDimension.FRIENDLINESS]
        adapted_response, friendliness_change = self._adapt_friendliness(
            adapted_response, friendliness_factor
        )
        adaptation_factors["friendliness"] = friendliness_change

        # Créativité
        creativity_factor = target_personality.personality_dimensions[PersonalityDimension.CREATIVITY]
        adapted_response, creativity_change = self._adapt_creativity(
            adapted_response, creativity_factor
        )
        adaptation_factors["creativity"] = creativity_change

        # Calcule le score de confiance
        confidence_score = target_personality.is_confident() if target_personality.sample_size > 0 else 0.5

        # Raisonnement
        reasoning = self._generate_adaptation_reasoning(
            original_tone, target_personality, adaptation_factors
        )

        return PersonalityAdaptation(
            original_tone=original_tone,
            adapted_tone=self._analyze_response_tone(adapted_response),
            adaptation_factors=adaptation_factors,
            confidence_score=confidence_score,
            reasoning=reasoning
        )

    def maintain_personality_consistency(
        self,
        conversation_context: List[Dict[str, Any]],
        current_personality: PersonalityProfile
    ) -> Dict[str, Any]:
        """Maintient la cohérence de personnalité dans la conversation.

        Args:
            conversation_context: Contexte de la conversation
            current_personality: Personnalité actuelle

        Returns:
            Recommandations pour maintenir la cohérence
        """
        if len(conversation_context) < 2:
            return {"consistency_score": 1.0, "recommendations": []}

        # Analyse la cohérence récente
        recent_messages = conversation_context[-10:]  # Derniers 10 messages
        consistency_scores = []

        for i in range(1, len(recent_messages)):
            prev_msg = recent_messages[i-1]
            curr_msg = recent_messages[i]

            if prev_msg.get("role") == "assistant" and curr_msg.get("role") == "assistant":
                # Compare les tons des réponses du bot
                prev_tone = self._analyze_response_tone(prev_msg["content"])
                curr_tone = self._analyze_response_tone(curr_msg["content"])

                consistency = self._calculate_tone_consistency(prev_tone, curr_tone)
                consistency_scores.append(consistency)

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0

        # Génère des recommandations
        recommendations = []
        if avg_consistency < 0.7:
            recommendations.append("Ton des réponses incohérent - adapter progressivement")
        if current_personality.sample_size < 10:
            recommendations.append("Profil insuffisamment établi - continuer l'apprentissage")

        return {
            "consistency_score": float(avg_consistency),
            "recommendations": recommendations,
            "analysis_period": len(recent_messages)
        }

    def learn_user_preferences(
        self,
        user_feedback: List[Dict[str, Any]],
        response_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apprend les préférences de l'utilisateur.

        Args:
            user_feedback: Retours de l'utilisateur
            response_history: Historique des réponses

        Returns:
            Apprentissages extraits
        """
        preferences = {
            "preferred_tones": {},
            "avoided_patterns": [],
            "response_preferences": {},
            "learning_confidence": 0.0
        }

        if not user_feedback or not response_history:
            return preferences

        # Analyse les retours positifs/négatifs
        positive_feedback = [f for f in user_feedback if f.get("sentiment") == "positive"]
        negative_feedback = [f for f in user_feedback if f.get("sentiment") == "negative"]

        # Identifie les patterns préférés
        if positive_feedback:
            preferences["preferred_tones"] = self._extract_preferred_patterns(
                positive_feedback, response_history
            )

        if negative_feedback:
            preferences["avoided_patterns"] = self._extract_avoided_patterns(
                negative_feedback, response_history
            )

        # Calcule la confiance de l'apprentissage
        total_feedback = len(user_feedback)
        if total_feedback > 0:
            preferences["learning_confidence"] = min(total_feedback / 20, 1.0)  # Max à 20 feedbacks

        return preferences

    def _initialize_style_analyzer(self) -> Dict[str, Any]:
        """Initialise l'analyseur de style."""
        return {
            "formal_markers": [
                r'\b(je vous|vous êtes|nous devons|il convient)\b',
                r'\b(monsieur|madame|docteur|professeur)\b',
                r'\b(veuillez|pourriez-vous|auriez-vous)\b'
            ],
            "casual_markers": [
                r'\b(salut|hey|yo|coucou)\b',
                r'\b(tu es|t\'es|on va)\b',
                r'\b(ok|cool|super|bien)\b'
            ],
            "technical_markers": [
                r'\b(algorithme|architecture|framework|api)\b',
                r'\b(déploiement|optimisation|scalabilité)\b',
                r'\b(débugger|compiler|parser)\b'
            ],
            "emotional_markers": {
                "positive": [r'\b(merci|super|excellent|génial)\b'],
                "negative": [r'\b(désolé|problème|erreur|bug)\b'],
                "friendly": [r'\b(aide|ensemble|équipe|communauté)\b']
            }
        }

    def _extract_linguistic_features(self, messages: List[str]) -> Dict[str, float]:
        """Extrait les caractéristiques linguistiques."""
        features = {
            "avg_sentence_length": 0.0,
            "vocabulary_richness": 0.0,
            "question_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "creativity_score": 0.0
        }

        if not messages:
            return features

        all_text = " ".join(messages)
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Longueur moyenne des phrases
        features["avg_sentence_length"] = np.mean([len(s.split()) for s in sentences])

        # Richesse du vocabulaire
        words = re.findall(r'\b\w+\b', all_text.lower())
        unique_words = set(words)
        features["vocabulary_richness"] = len(unique_words) / len(words) if words else 0

        # Ratio de questions
        question_count = sum(1 for msg in messages if '?' in msg)
        features["question_ratio"] = question_count / len(messages)

        # Ratio d'exclamations
        exclamation_count = sum(1 for msg in messages if '!' in msg)
        features["exclamation_ratio"] = exclamation_count / len(messages)

        # Score de créativité (basé sur la diversité des structures)
        sentence_structures = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        features["creativity_score"] = np.std(sentence_structures) / np.mean(sentence_structures) if sentence_structures else 0

        return features

    def _extract_emotional_indicators(self, messages: List[str]) -> Dict[str, float]:
        """Extrait les indicateurs émotionnels."""
        indicators = {
            "friendliness": 0.0,
            "urgency": 0.0,
            "politeness": 0.0,
            "enthusiasm": 0.0
        }

        if not messages:
            return indicators

        all_text = " ".join(messages).lower()

        # Amicalité
        friendly_words = ["merci", "aide", "ensemble", "super", "génial", "cool"]
        friendly_count = sum(1 for word in friendly_words if word in all_text)
        indicators["friendliness"] = friendly_count / len(messages)

        # Urgence
        urgent_words = ["urgent", "vite", "rapidement", "asap", "immédiatement"]
        urgent_count = sum(1 for word in urgent_words if word in all_text)
        indicators["urgency"] = urgent_count / len(messages)

        # Politesse
        polite_words = ["s'il vous plaît", "pourriez-vous", "veuillez", "merci"]
        polite_count = sum(1 for word in polite_words if word in all_text)
        indicators["politeness"] = polite_count / len(messages)

        # Enthousiasme
        enthusiastic_patterns = [r'!+', r'\b(wow|super|génial|excellent)\b']
        enthusiasm_score = 0
        for pattern in enthusiastic_patterns:
            enthusiasm_score += len(re.findall(pattern, all_text))
        indicators["enthusiasm"] = enthusiasm_score / len(messages)

        return indicators

    def _extract_complexity_metrics(self, messages: List[str]) -> Dict[str, float]:
        """Extrait les métriques de complexité."""
        metrics = {
            "avg_complexity": 0.0,
            "technical_terms_ratio": 0.0,
            "sentence_complexity": 0.0
        }

        if not messages:
            return metrics

        all_text = " ".join(messages).lower()

        # Termes techniques
        technical_terms = [
            "algorithme", "architecture", "framework", "api", "déploiement",
            "optimisation", "scalabilité", "debug", "compile", "parse"
        ]
        technical_count = sum(1 for term in technical_terms if term in all_text)
        metrics["technical_terms_ratio"] = technical_count / len(messages)

        # Complexité des phrases
        sentences = re.split(r'[.!?]+', all_text)
        complexities = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                # Complexité basée sur la longueur et la diversité
                complexity = len(words) * (len(set(words)) / len(words))
                complexities.append(complexity)

        metrics["sentence_complexity"] = np.mean(complexities) if complexities else 0
        metrics["avg_complexity"] = (metrics["technical_terms_ratio"] + metrics["sentence_complexity"]) / 2

        return metrics

    def _calculate_formality_score(
        self,
        linguistic_features: Dict[str, float],
        emotional_indicators: Dict[str, float]
    ) -> float:
        """Calcule le score de formalité."""
        # Combinaison des caractéristiques
        formality_factors = [
            emotional_indicators.get("politeness", 0) * 0.4,
            (1 - emotional_indicators.get("friendliness", 0)) * 0.3,
            linguistic_features.get("vocabulary_richness", 0) * 0.3
        ]

        return np.mean(formality_factors)

    def _determine_communication_style(
        self,
        linguistic_features: Dict[str, float],
        emotional_indicators: Dict[str, float],
        complexity_metrics: Dict[str, float],
        formality_score: float
    ) -> Tuple[CommunicationStyle, float]:
        """Détermine le style de communication."""
        scores = {
            CommunicationStyle.FORMAL: formality_score,
            CommunicationStyle.CASUAL: 1 - formality_score,
            CommunicationStyle.TECHNICAL: complexity_metrics.get("technical_terms_ratio", 0),
            CommunicationStyle.SIMPLE: 1 - complexity_metrics.get("avg_complexity", 0),
            CommunicationStyle.FRIENDLY: emotional_indicators.get("friendliness", 0),
            CommunicationStyle.PROFESSIONAL: formality_score * 0.8 + emotional_indicators.get("politeness", 0) * 0.2,
            CommunicationStyle.CREATIVE: linguistic_features.get("creativity_score", 0),
            CommunicationStyle.DIRECT: 1 - emotional_indicators.get("politeness", 0)
        }

        best_style = max(scores, key=scores.get)
        confidence = scores[best_style]

        return best_style, confidence

    def _analyze_response_tone(self, response: str) -> str:
        """Analyse le ton d'une réponse."""
        response_lower = response.lower()

        # Analyse simple du ton
        if any(word in response_lower for word in ["cher", "monsieur", "madame", "veuillez"]):
            return "formal"
        elif any(word in response_lower for word in ["salut", "hey", "cool", "super"]):
            return "casual"
        elif any(word in response_lower for word in ["algorithme", "technique", "optimisation"]):
            return "technical"
        else:
            return "neutral"

    def _adapt_formality(self, response: str, formality_factor: float) -> Tuple[str, float]:
        """Adapte la formalité de la réponse."""
        if formality_factor > 0.7:  # Très formel
            adaptations = {
                "tu ": "vous ",
                "t'": "vous ",
                "je pense": "je pense",
                "salut": "bonjour",
                "hey": "bonjour"
            }
        elif formality_factor < 0.3:  # Très casual
            adaptations = {
                "vous ": "tu ",
                "bonjour": "salut",
                "monsieur": "",
                "madame": ""
            }
        else:
            return response, 0.0

        adapted = response
        change_count = 0
        for old, new in adaptations.items():
            if old in adapted.lower():
                adapted = re.sub(re.escape(old), new, adapted, flags=re.IGNORECASE)
                change_count += 1

        return adapted, change_count / len(response.split()) if response.split() else 0

    def _adapt_complexity(self, response: str, complexity_factor: float) -> Tuple[str, float]:
        """Adapte la complexité de la réponse."""
        # Pour l'instant, adaptation simple
        if complexity_factor > 0.7:
            # Ajoute des termes techniques si approprié
            return response, 0.0
        elif complexity_factor < 0.3:
            # Simplifie le langage
            return response, 0.0
        return response, 0.0

    def _adapt_friendliness(self, response: str, friendliness_factor: float) -> Tuple[str, float]:
        """Adapte l'amicalité de la réponse."""
        if friendliness_factor > 0.7:
            # Rend plus amical
            friendly_additions = ["", " 😊", " Avec plaisir !"]
            addition = np.random.choice(friendly_additions)
            return response + addition, 0.1
        elif friendliness_factor < 0.3:
            # Rend plus professionnel
            return response, 0.0
        return response, 0.0

    def _adapt_creativity(self, response: str, creativity_factor: float) -> Tuple[str, float]:
        """Adapte la créativité de la réponse."""
        # Pour l'instant, pas d'adaptation spécifique
        return response, 0.0

    def _calculate_tone_consistency(self, tone1: str, tone2: str) -> float:
        """Calcule la cohérence entre deux tons."""
        if tone1 == tone2:
            return 1.0
        elif abs(ord(tone1[0]) - ord(tone2[0])) <= 2:  # Tons similaires
            return 0.7
        else:
            return 0.3

    def _generate_adaptation_reasoning(
        self,
        original_tone: str,
        personality: PersonalityProfile,
        factors: Dict[str, float]
    ) -> str:
        """Génère le raisonnement pour l'adaptation."""
        reasoning_parts = []

        if factors.get("formality", 0) > 0:
            formality = personality.personality_dimensions[PersonalityDimension.FORMALITY]
            if formality > 0.7:
                reasoning_parts.append("Ton adapté pour être plus formel")
            elif formality < 0.3:
                reasoning_parts.append("Ton adapté pour être plus casual")

        if factors.get("friendliness", 0) > 0:
            friendliness = personality.personality_dimensions[PersonalityDimension.FRIENDLINESS]
            if friendliness > 0.7:
                reasoning_parts.append("Ajout d'éléments amicaux")

        return " ; ".join(reasoning_parts) if reasoning_parts else "Aucune adaptation majeure nécessaire"

    def _extract_preferred_patterns(
        self,
        positive_feedback: List[Dict[str, Any]],
        response_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extrait les patterns préférés."""
        # Analyse simple pour l'instant
        return {"casual": 0.6, "helpful": 0.8}

    def _extract_avoided_patterns(
        self,
        negative_feedback: List[Dict[str, Any]],
        response_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Extrait les patterns à éviter."""
        # Analyse simple pour l'instant
        return ["too_technical", "too_formal"]