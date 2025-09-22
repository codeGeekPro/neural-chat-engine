"""Système d'analyse des émotions multi-langue."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline
)

from .emotion_types import EmotionCategory, EmotionProfile, EmotionState


logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """Système d'analyse des émotions dans les conversations."""

    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ) -> None:
        """Initialise l'analyseur d'émotions.
        
        Args:
            model_name: Nom du modèle HuggingFace à utiliser
            cache_dir: Dossier de cache pour les modèles
            device: Périphérique d'inférence ('cpu' ou 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Chargement du pipeline de classification
        self.classifier = self._load_emotion_classifier()
        
        # Cache des profils émotionnels
        self._emotion_profiles: Dict[str, EmotionProfile] = {}
        
        # Mapping des émotions du modèle vers nos catégories
        self._emotion_mapping = self._load_emotion_mapping()
        
        logger.info(
            f"EmotionAnalyzer initialisé avec {model_name} sur {device}"
        )

    def _load_emotion_classifier(self) -> Pipeline:
        """Charge le modèle de classification des émotions."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        ).to(self.device)
        
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            top_k=None  # Retourne toutes les classes
        )

    def _load_emotion_mapping(self) -> Dict[str, EmotionCategory]:
        """Charge le mapping des labels du modèle vers nos catégories."""
        # Exemple de mapping pour go_emotions
        return {
            "joy": EmotionCategory.JOY,
            "sadness": EmotionCategory.SADNESS,
            "anger": EmotionCategory.ANGER,
            "fear": EmotionCategory.FEAR,
            "surprise": EmotionCategory.SURPRISE,
            "disgust": EmotionCategory.DISGUST,
            "trust": EmotionCategory.TRUST,
            "anticipation": EmotionCategory.ANTICIPATION,
            "love": EmotionCategory.LOVE,
            "guilt": EmotionCategory.GUILT,
            "pride": EmotionCategory.PRIDE,
            "shame": EmotionCategory.SHAME,
            "anxiety": EmotionCategory.ANXIETY,
            "neutral": EmotionCategory.NEUTRAL
        }

    def analyze_message(
        self,
        text: str,
        context: Optional[List[str]] = None
    ) -> EmotionState:
        """Analyse l'état émotionnel d'un message.
        
        Args:
            text: Le message à analyser
            context: Messages de contexte optionnels

        Returns:
            L'état émotionnel détecté
        """
        # Analyse brute du modèle
        raw_emotions = self.classifier(text)[0]
        
        # Normalise les scores
        scores = {
            self._emotion_mapping.get(e["label"], EmotionCategory.UNKNOWN): e["score"]
            for e in raw_emotions
        }
        
        # Trouve l'émotion dominante
        primary_emotion = max(scores.items(), key=lambda x: x[1])[0]
        max_intensity = max(scores.values())
        
        return EmotionState(
            primary_emotion=primary_emotion,
            intensity=max_intensity,
            emotion_scores=scores,
            timestamp=datetime.now(),
            confidence=max_intensity,
            context_window=context
        )

    def analyze_conversation(
        self,
        messages: List[Tuple[str, str]],
        user_id: Optional[str] = None
    ) -> List[EmotionState]:
        """Analyse une conversation complète.
        
        Args:
            messages: Liste de tuples (role, contenu)
            user_id: ID utilisateur optionnel pour le profil

        Returns:
            Liste des états émotionnels détectés
        """
        states: List[EmotionState] = []
        context_window: List[str] = []
        
        for role, content in messages:
            # On analyse principalement les messages utilisateur
            if role.lower() == "user":
                # Utilise les n derniers messages comme contexte
                state = self.analyze_message(content, context=context_window[-3:])
                states.append(state)
                
                # Met à jour le profil si un ID est fourni
                if user_id:
                    self._update_user_profile(user_id, state)
            
            # Maintient la fenêtre de contexte
            context_window.append(content)
            if len(context_window) > 5:  # Taille max de contexte
                context_window.pop(0)
                
        return states

    def get_user_profile(self, user_id: str) -> EmotionProfile:
        """Récupère le profil émotionnel d'un utilisateur."""
        if user_id not in self._emotion_profiles:
            self._emotion_profiles[user_id] = EmotionProfile()
        return self._emotion_profiles[user_id]

    def _update_user_profile(self, user_id: str, state: EmotionState) -> None:
        """Met à jour le profil émotionnel avec un nouvel état."""
        profile = self.get_user_profile(user_id)
        profile.add_emotion_state(state)

    def save_profiles(self, file_path: str) -> None:
        """Sauvegarde les profils émotionnels sur disque."""
        data = {}
        for user_id, profile in self._emotion_profiles.items():
            data[user_id] = {
                "emotion_frequencies": profile.emotion_frequencies,
                "avg_intensities": profile.avg_intensities,
                "transitions": profile.common_transitions
            }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_profiles(self, file_path: str) -> None:
        """Charge les profils émotionnels depuis le disque."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for user_id, profile_data in data.items():
            profile = EmotionProfile()
            profile.emotion_frequencies = {
                EmotionCategory(k): v 
                for k, v in profile_data["emotion_frequencies"].items()
            }
            profile.avg_intensities = {
                EmotionCategory(k): v 
                for k, v in profile_data["avg_intensities"].items()
            }
            profile.common_transitions = {
                EmotionCategory(k1): {
                    EmotionCategory(k2): v 
                    for k2, v in v1.items()
                }
                for k1, v1 in profile_data["transitions"].items()
            }
            self._emotion_profiles[user_id] = profile