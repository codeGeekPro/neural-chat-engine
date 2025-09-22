"""Générateur de réponses avancé avec RAG et contrôle de personnalité."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer
)

from .response_types import (
    GeneratedResponse,
    PersonalityProfile,
    PersonalityTrait,
    RAGResult,
    ResponseFormat
)


logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Générateur de réponses avancé avec RAG."""

    def __init__(
        self,
        base_model: str = "google/flan-t5-large",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        personality: Optional[PersonalityProfile] = None
    ) -> None:
        """Initialise le générateur.
        
        Args:
            base_model: Modèle de base (T5, GPT, etc.)
            cache_dir: Dossier de cache pour les modèles
            device: Périphérique d'inférence ('cpu' ou 'cuda')
            personality: Profil de personnalité optionnel
        """
        self.model_name = base_model
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Charge les modèles
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # Système RAG
        self.retriever: Optional[SentenceTransformer] = None
        self.vector_store: Any = None  # Type concret dépend de l'implémentation
        
        # Personnalité
        self.personality = personality or PersonalityProfile()
        
        # Cache de prompts
        self._prompt_templates: Dict[str, str] = self._load_prompt_templates()
        
        logger.info(
            f"ResponseGenerator initialisé avec {base_model} sur {device}"
        )

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Charge le tokenizer adapté au modèle."""
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )

    def _load_model(self) -> PreTrainedModel:
        """Charge le modèle de génération."""
        if "t5" in self.model_name.lower():
            model_class = AutoModelForSeq2SeqLM
        else:
            model_class = AutoModelForCausalLM
            
        return model_class.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            device_map=self.device
        )

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Charge les templates de prompts."""
        # TODO: Charger depuis un fichier de config
        return {
            "default": (
                "Contexte: {context}\n\n"
                "Historique: {history}\n\n"
                "Utilisateur: {user_input}\n\n"
                "Assistant: "
            ),
            "rag": (
                "Contexte pertinent:\n{context}\n\n"
                "Historique de la conversation:\n{history}\n\n"
                "Profil utilisateur:\n{user_profile}\n\n"
                "Question: {user_input}\n\n"
                "Instructions: Utilise le contexte ci-dessus pour générer une "
                "réponse informative et cohérente. La réponse doit être:\n"
                "- Précise et factuelle\n"
                "- Adaptée au niveau de l'utilisateur\n"
                "- Dans un style {style}\n\n"
                "Réponse:"
            ),
            "personality": (
                "Tu es un assistant avec la personnalité suivante:\n"
                "- Ouverture: {openness:+.1f}\n"
                "- Conscience: {conscientiousness:+.1f}\n"
                "- Extraversion: {extraversion:+.1f}\n"
                "- Agréabilité: {agreeableness:+.1f}\n"
                "- Neurotisme: {neuroticism:+.1f}\n\n"
                "Style:\n"
                "- Formalité: {formality:.1%}\n"
                "- Humour: {humor:.1%}\n"
                "- Empathie: {empathy:.1%}\n"
                "- Expertise: {expertise:.1%}\n"
                "- Proactivité: {proactivity:.1%}\n\n"
                "Génère une réponse avec cette personnalité:\n"
                "{base_response}"
            )
        }

    def setup_rag_system(
        self,
        vector_store: Any,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """Configure le système RAG.
        
        Args:
            vector_store: Base de données vectorielle
            retriever_model: Modèle d'embeddings
        """
        self.vector_store = vector_store
        self.retriever = SentenceTransformer(
            retriever_model,
            cache_folder=str(self.cache_dir) if self.cache_dir else None,
            device=self.device
        )
        logger.info(f"Système RAG configuré avec {retriever_model}")

    def _retrieve_context(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.6
    ) -> RAGResult:
        """Recherche le contexte pertinent via RAG."""
        if not self.retriever or not self.vector_store:
            raise RuntimeError("Système RAG non configuré")
            
        # Calcule l'embedding de la requête
        query_vector = self.retriever.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Recherche les documents similaires
        results = self.vector_store.search(
            query_vector.cpu().numpy(),
            k=k,
            min_score=min_score
        )
        
        return RAGResult(
            documents=results["documents"],
            scores=results["scores"],
            query_vector=query_vector.cpu().numpy().tolist()
        )

    def generate_response(
        self,
        user_input: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        response_format: ResponseFormat = ResponseFormat.TEXT,
        max_length: int = 512
    ) -> GeneratedResponse:
        """Génère une réponse adaptée.
        
        Args:
            user_input: Message utilisateur
            conversation_context: Historique de conversation
            user_profile: Profil utilisateur
            response_format: Format de sortie
            max_length: Longueur maximale de la réponse

        Returns:
            Réponse générée avec métadonnées
        """
        # Récupère le contexte via RAG
        if self.retriever:
            rag_result = self._retrieve_context(user_input)
            context = rag_result.get_formatted_context()
        else:
            context = ""
            rag_result = None
            
        # Construit l'historique formaté
        history = ""
        if conversation_context:
            history = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in conversation_context[-5:]  # 5 derniers messages
            )
            
        # Construit le prompt
        prompt_template = self._prompt_templates["rag" if context else "default"]
        prompt = prompt_template.format(
            context=context,
            history=history,
            user_profile=json.dumps(user_profile) if user_profile else "",
            user_input=user_input,
            style=self._get_personality_style()
        )
        
        # Génère la réponse brute
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
        response_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Adapte la personnalité
        response_text = self.adapt_personality(response_text)
        
        # Valide et formate la réponse
        quality_metrics = self.validate_response_quality(
            response_text,
            context=prompt
        )
        
        return GeneratedResponse(
            content=response_text,
            format=response_format,
            metadata={
                "rag_result": rag_result._asdict() if rag_result else None,
                "personality": self.personality.dict(),
                "prompt_length": len(prompt),
                "response_length": len(response_text)
            },
            quality_metrics=quality_metrics
        )

    def adapt_personality(
        self,
        response: str,
        personality_override: Optional[Dict[PersonalityTrait, float]] = None
    ) -> str:
        """Adapte la réponse selon la personnalité.
        
        Args:
            response: Réponse brute
            personality_override: Surcharge de traits

        Returns:
            Réponse adaptée
        """
        # Applique les surcharges temporaires
        original_traits = {}
        if personality_override:
            for trait, value in personality_override.items():
                original_traits[trait] = getattr(self.personality, trait.value)
                self.personality.adjust_trait(trait, value, weight=1.0)
                
        # Formate avec le template de personnalité
        personality_prompt = self._prompt_templates["personality"].format(
            base_response=response,
            **self.personality.dict()
        )
        
        # Régénère avec la personnalité
        inputs = self.tokenizer(
            personality_prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(response) * 2,  # Marge pour l'adaptation
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )
            
        adapted_response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Restaure les traits originaux
        for trait, value in original_traits.items():
            self.personality.adjust_trait(trait, value, weight=1.0)
            
        return adapted_response

    def validate_response_quality(
        self,
        response: str,
        context: Optional[str] = None
    ) -> Dict[str, float]:
        """Vérifie la qualité de la réponse.
        
        Args:
            response: Réponse à valider
            context: Contexte de génération

        Returns:
            Métriques de qualité
        """
        metrics = {
            "length_score": min(len(response.split()) / 100, 1.0),
            "coherence_score": self._estimate_coherence(response, context),
            "toxicity_score": 0.0  # TODO: Intégrer un détecteur
        }
        
        if context:
            metrics["relevance_score"] = self._estimate_relevance(
                response,
                context
            )
            
        return metrics

    def _estimate_coherence(
        self,
        text: str,
        context: Optional[str] = None
    ) -> float:
        """Estime la cohérence du texte (heuristique simple)."""
        # TODO: Utiliser un vrai modèle de cohérence
        words = text.split()
        if len(words) < 3:
            return 0.0
            
        # Ratio mots uniques / total
        unique_ratio = len(set(words)) / len(words)
        
        # Pénalise les ratios extrêmes
        if unique_ratio < 0.2:  # Trop répétitif
            return 0.3
        elif unique_ratio > 0.95:  # Probablement incohérent
            return 0.5
            
        return min(unique_ratio + 0.3, 1.0)

    def _estimate_relevance(self, response: str, context: str) -> float:
        """Estime la pertinence par rapport au contexte."""
        if not self.retriever:
            return 0.8  # Valeur par défaut optimiste
            
        # Compare les embeddings
        embeddings = self.retriever.encode(
            [response, context],
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        similarity = torch.cosine_similarity(
            embeddings[0],
            embeddings[1],
            dim=0
        )
        
        return float(similarity)

    def _get_personality_style(self) -> str:
        """Détermine le style basé sur la personnalité."""
        if self.personality.formality > 0.7:
            return "formel et professionnel"
        elif self.personality.formality < 0.3:
            return "décontracté et familier"
        else:
            return "naturel et équilibré"