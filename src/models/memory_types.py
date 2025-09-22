"""Types et structures pour le système de mémoire conversationnelle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data as GraphData


class MemoryType(str, Enum):
    """Types de mémoire supportés."""
    
    SHORT_TERM = "short_term"  # Mémoire de travail (contexte immédiat)
    LONG_TERM = "long_term"  # Mémoire sémantique et épisodique
    COMPRESSED = "compressed"  # Mémoire compressée/résumée
    RELATIONSHIP = "relationship"  # Graphe de relations


class EntityType(str, Enum):
    """Types d'entités dans le graphe de mémoire."""
    
    USER = "user"  # Utilisateur principal
    PERSON = "person"  # Autre personne mentionnée
    CONCEPT = "concept"  # Concept ou idée abstraite
    TOPIC = "topic"  # Sujet de conversation
    FACT = "fact"  # Information factuelle
    EVENT = "event"  # Événement temporel
    PREFERENCE = "preference"  # Préférence utilisateur


class RelationType(str, Enum):
    """Types de relations entre entités."""
    
    KNOWS = "knows"  # Connaissance entre personnes
    INTERESTED_IN = "interested_in"  # Intérêt pour un sujet/concept
    EXPERIENCED = "experienced"  # A vécu un événement
    RELATED_TO = "related_to"  # Relation générique
    IMPLIES = "implies"  # Relation d'implication
    PART_OF = "part_of"  # Relation de composition
    PRECEDES = "precedes"  # Relation temporelle


@dataclass
class MemoryEmbedding:
    """Représentation vectorielle d'un élément mémorisé."""
    
    vector: np.ndarray  # Embedding brut
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_hours(self) -> float:
        """Âge de l'embedding en heures."""
        delta = datetime.now() - self.timestamp
        return delta.total_seconds() / 3600


@dataclass
class ConversationMemoryItem:
    """Élément unitaire de mémoire conversationnelle."""
    
    content: str  # Contenu textuel
    embedding: MemoryEmbedding  # Représentation vectorielle
    memory_type: MemoryType  # Type de mémoire
    importance_score: float = 1.0  # Score d'importance [0, 1]
    references: Set[str] = field(default_factory=set)  # Liens vers d'autres items
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_compress(self, threshold: float = 0.3) -> bool:
        """Détermine si l'item doit être compressé."""
        if self.memory_type == MemoryType.COMPRESSED:
            return False
        
        age_factor = min(self.embedding.age_hours / 24.0, 1.0)
        memory_score = self.importance_score * (1.0 - age_factor)
        
        return memory_score < threshold


@dataclass
class MemoryBatch:
    """Lot de souvenirs pour traitement groupé."""
    
    items: List[ConversationMemoryItem]
    embeddings: torch.Tensor
    types: List[MemoryType]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphEntity:
    """Entité dans le graphe de connaissances."""
    
    id: str  # Identifiant unique
    type: EntityType  # Type d'entité
    name: str  # Nom/label lisible
    embedding: Optional[np.ndarray] = None  # Représentation vectorielle
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'entité en dictionnaire."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "attributes": self.attributes
        }


@dataclass
class KnowledgeGraphRelation:
    """Relation dans le graphe de connaissances."""
    
    source_id: str  # ID entité source
    target_id: str  # ID entité cible
    type: RelationType  # Type de relation
    weight: float = 1.0  # Poids/force de la relation
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la relation en dictionnaire."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.type.value,
            "weight": self.weight,
            "attributes": self.attributes
        }


class MemoryGraph:
    """Graphe de mémoire basé sur PyTorch Geometric."""
    
    def __init__(self) -> None:
        self.entities: Dict[str, KnowledgeGraphEntity] = {}
        self.relations: List[KnowledgeGraphRelation] = []
        self._graph_data: Optional[GraphData] = None
        self._needs_update: bool = True
    
    def add_entity(self, entity: KnowledgeGraphEntity) -> None:
        """Ajoute une entité au graphe."""
        self.entities[entity.id] = entity
        self._needs_update = True
    
    def add_relation(self, relation: KnowledgeGraphRelation) -> None:
        """Ajoute une relation au graphe."""
        if (relation.source_id in self.entities and 
            relation.target_id in self.entities):
            self.relations.append(relation)
            self._needs_update = True
    
    def get_entity_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None
    ) -> List[KnowledgeGraphEntity]:
        """Récupère les voisins d'une entité."""
        neighbors = []
        for rel in self.relations:
            if rel.source_id == entity_id:
                if (relation_type is None or 
                    rel.type == relation_type):
                    neighbors.append(self.entities[rel.target_id])
            elif rel.target_id == entity_id:
                if (relation_type is None or 
                    rel.type == relation_type):
                    neighbors.append(self.entities[rel.source_id])
        return neighbors
    
    def to_pytorch_geometric(self) -> GraphData:
        """Convertit en format PyTorch Geometric."""
        if not self._needs_update and self._graph_data is not None:
            return self._graph_data
            
        # Crée les tenseurs de noeuds et d'arêtes
        node_features = []
        edge_index = []
        edge_attr = []
        
        # Map des IDs vers indices
        id_to_idx = {eid: i for i, eid in enumerate(self.entities)}
        
        # Noeuds
        for entity in self.entities.values():
            if entity.embedding is not None:
                node_features.append(entity.embedding)
            else:
                # Embedding par défaut de même dimension
                node_features.append(
                    np.zeros_like(
                        next(e.embedding for e in self.entities.values() 
                             if e.embedding is not None)
                    )
                )
        
        # Arêtes
        for rel in self.relations:
            src_idx = id_to_idx[rel.source_id]
            tgt_idx = id_to_idx[rel.target_id]
            edge_index.extend([[src_idx, tgt_idx]])
            edge_attr.append([
                rel.weight,
                len(RelationType),  # One-hot pour le type
                len(rel.attributes)  # Nombre d'attributs
            ])
        
        # Crée le graphe PyTorch
        self._graph_data = GraphData(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        )
        
        self._needs_update = False
        return self._graph_data


@dataclass
class UserProfile:
    """Profil utilisateur basé sur l'historique des conversations."""
    
    user_id: str
    interests: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    knowledge_areas: Dict[str, float] = field(default_factory=dict)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    interaction_history: Dict[str, int] = field(default_factory=dict)
    
    def update_from_conversation(
        self,
        content: str,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """Met à jour le profil depuis une conversation."""
        # TODO: Implémenter l'analyse et mise à jour
        pass
    
    def get_profile_embedding(self) -> np.ndarray:
        """Calcule un embedding représentatif du profil."""
        # TODO: Implémenter la fusion des caractéristiques
        return np.zeros(768)  # Placeholder
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le profil en dictionnaire."""
        return {
            "user_id": self.user_id,
            "interests": self.interests,
            "preferences": self.preferences,
            "knowledge_areas": self.knowledge_areas,
            "personality_traits": self.personality_traits,
            "interaction_stats": self.interaction_history
        }