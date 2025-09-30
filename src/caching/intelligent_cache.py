"""
Système de cache intelligent pour les réponses et les embeddings.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import torch
from sentence_transformers import SentenceTransformer
import redis
from collections import OrderedDict
import logging
import json

@dataclass
class CacheEntry:
    """Structure de données pour une entrée de cache."""
    content: Any
    embedding: np.ndarray
    timestamp: datetime
    access_count: int
    memory_size: int

class IntelligentCache:
    def __init__(
        self,
        max_size_mb: float = 1024,
        similarity_threshold: float = 0.85,
        ttl_hours: int = 24,
        distributed: bool = False,
        redis_url: Optional[str] = None
    ):
        """
        Initialise le système de cache intelligent.
        
        Args:
            max_size_mb: Taille maximale du cache en MB
            similarity_threshold: Seuil de similarité sémantique
            ttl_hours: Durée de vie des entrées en heures
            distributed: Utiliser un cache distribué avec Redis
            redis_url: URL de connexion Redis si distribué
        """
        self.max_size = max_size_mb * 1024 * 1024  # Conversion en bytes
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
        self.distributed = distributed
        
        # Initialisation du modèle d'embedding
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cache local
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        
        # Cache distribué
        self.redis_client = None
        if distributed and redis_url:
            self.redis_client = redis.from_url(redis_url)
        
        self.logger = logging.getLogger(__name__)
        
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Calcule l'embedding d'un texte."""
        return self.embedding_model.encode([text])[0]
    
    def _compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Calcule la similarité cosinus entre deux embeddings."""
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def _get_memory_size(self, obj: Any) -> int:
        """Estime la taille mémoire d'un objet."""
        if isinstance(obj, (str, bytes)):
            return len(obj)
        return len(json.dumps(obj))

    def _cleanup_expired(self) -> None:
        """Supprime les entrées expirées du cache."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry.timestamp > self.ttl
        ]
        for key in expired_keys:
            self._remove_entry(key)

    def _remove_least_valuable(self) -> None:
        """
        Supprime les entrées les moins valorisées selon un score composite
        basé sur la fréquence d'accès et l'âge.
        """
        if not self.cache:
            return

        now = datetime.now()
        scores = {
            key: (entry.access_count / 
                  (now - entry.timestamp).total_seconds()) * entry.memory_size
            for key, entry in self.cache.items()
        }
        
        least_valuable = min(scores.items(), key=lambda x: x[1])[0]
        self._remove_entry(least_valuable)

    def _remove_entry(self, key: str) -> None:
        """Supprime une entrée du cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.memory_size
            if self.distributed and self.redis_client:
                self.redis_client.delete(key)

    def get(
        self, 
        query: str, 
        semantic_search: bool = True
    ) -> Optional[Tuple[Any, float]]:
        """
        Récupère une entrée du cache.
        
        Args:
            query: Clé ou texte de requête
            semantic_search: Utiliser la recherche sémantique
            
        Returns:
            Tuple (contenu, score_similarité) ou None si non trouvé
        """
        self._cleanup_expired()
        
        # Recherche exacte
        if query in self.cache:
            entry = self.cache[query]
            entry.access_count += 1
            return entry.content, 1.0
            
        # Recherche sémantique
        if semantic_search:
            query_embedding = self._compute_embedding(query)
            best_match = None
            best_similarity = 0
            
            for key, entry in self.cache.items():
                similarity = self._compute_similarity(query_embedding, entry.embedding)
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
            
            if best_match:
                best_match.access_count += 1
                return best_match.content, best_similarity
        
        # Recherche dans le cache distribué
        if self.distributed and self.redis_client:
            redis_value = self.redis_client.get(query)
            if redis_value:
                entry_dict = json.loads(redis_value)
                entry = CacheEntry(**entry_dict)
                self.set(query, entry.content)  # Mise en cache local
                return entry.content, 1.0
                
        return None, 0.0

    def set(self, key: str, value: Any) -> None:
        """
        Ajoute ou met à jour une entrée dans le cache.
        
        Args:
            key: Clé de l'entrée
            value: Valeur à mettre en cache
        """
        # Calcul de la taille et de l'embedding
        memory_size = self._get_memory_size(value)
        embedding = self._compute_embedding(key)
        
        # Vérification de l'espace disponible
        while self.current_size + memory_size > self.max_size:
            self._remove_least_valuable()
        
        # Création de l'entrée
        entry = CacheEntry(
            content=value,
            embedding=embedding,
            timestamp=datetime.now(),
            access_count=1,
            memory_size=memory_size
        )
        
        # Mise à jour du cache local
        if key in self.cache:
            self.current_size -= self.cache[key].memory_size
        self.cache[key] = entry
        self.current_size += memory_size
        
        # Mise à jour du cache distribué
        if self.distributed and self.redis_client:
            entry_dict = {
                'content': value,
                'embedding': embedding.tolist(),
                'timestamp': entry.timestamp.isoformat(),
                'access_count': entry.access_count,
                'memory_size': memory_size
            }
            self.redis_client.set(
                key,
                json.dumps(entry_dict),
                ex=int(self.ttl.total_seconds())
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            'total_entries': len(self.cache),
            'current_size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size / (1024 * 1024),
            'hit_rate': 0.0,
            'memory_usage_percent': (self.current_size / self.max_size) * 100,
            'average_entry_size_kb': 0.0,
            'oldest_entry_age': 0,
            'distributed_mode': self.distributed
        }
        
        if self.cache:
            total_size = sum(entry.memory_size for entry in self.cache.values())
            stats['average_entry_size_kb'] = (total_size / len(self.cache)) / 1024
            
            oldest = min(entry.timestamp for entry in self.cache.values())
            stats['oldest_entry_age'] = (datetime.now() - oldest).total_seconds() / 3600
            
        return stats

    def clear(self) -> None:
        """Vide le cache."""
        self.cache.clear()
        self.current_size = 0
        if self.distributed and self.redis_client:
            self.redis_client.flushdb()