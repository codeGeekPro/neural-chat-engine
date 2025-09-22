"""Système de mémoire conversationnelle basé sur les GNNs."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GATConv, GCNConv
from torch_scatter import scatter_mean

from .memory_types import (
    ConversationMemoryItem,
    EntityType,
    KnowledgeGraphEntity,
    KnowledgeGraphRelation,
    MemoryBatch,
    MemoryEmbedding,
    MemoryGraph,
    MemoryType,
    RelationType,
    UserProfile
)


logger = logging.getLogger(__name__)


class MemoryGNN(nn.Module):
    """Réseau de neurones pour le traitement du graphe de mémoire."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()
        
        # Couches de graphe
        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=0.2
        )
        self.gat2 = GATConv(
            hidden_dim * num_heads,
            output_dim,
            heads=1,
            concat=False,
            dropout=0.2
        )
        
        # MLP pour la fusion finale
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        """Passe avant dans le réseau."""
        x, edge_index = data.x, data.edge_index
        
        # Attention multi-tête
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Seconde couche d'attention
        x = self.gat2(x, edge_index)
        
        # Fusion et normalisation
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=-1)


class ConversationMemory:
    """Système de mémoire conversationnelle avec GNN."""

    def __init__(
        self,
        max_context_length: int = 4096,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """Initialise le système de mémoire.
        
        Args:
            max_context_length: Longueur maximale du contexte
            embedding_model: Modèle pour les embeddings
            device: Périphérique de calcul
            cache_dir: Dossier de cache pour les modèles
        """
        self.max_context_length = max_context_length
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Modèles
        self.embedding_model = SentenceTransformer(
            embedding_model,
            cache_folder=str(self.cache_dir) if self.cache_dir else None,
            device=self.device
        )
        self.gnn = MemoryGNN(
            input_dim=self.embedding_model.get_sentence_embedding_dimension(),
            device=self.device
        )
        
        # Stockage mémoire
        self.short_term: List[ConversationMemoryItem] = []
        self.long_term: List[ConversationMemoryItem] = []
        self.compressed: List[ConversationMemoryItem] = []
        
        # Graphe de connaissances
        self.graph = MemoryGraph()
        
        # Cache de profils utilisateurs
        self.user_profiles: Dict[str, UserProfile] = {}
        
        logger.info(
            f"ConversationMemory initialisé avec contexte max={max_context_length}"
        )

    def store_conversation_turn(
        self,
        user_input: str,
        bot_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stocke un tour de conversation.
        
        Args:
            user_input: Message utilisateur
            bot_response: Réponse du bot
            metadata: Métadonnées additionnelles
        """
        # Crée les embeddings
        texts = [user_input, bot_response]
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Crée les items de mémoire
        timestamp = datetime.now()
        items = []
        
        for text, emb in zip(texts, embeddings):
            item = ConversationMemoryItem(
                content=text,
                embedding=MemoryEmbedding(
                    vector=emb.cpu().numpy(),
                    timestamp=timestamp,
                    metadata=metadata or {}
                ),
                memory_type=MemoryType.SHORT_TERM,
                importance_score=self._estimate_importance(text, emb)
            )
            items.append(item)
            
        # Ajoute à la mémoire court terme
        self.short_term.extend(items)
        
        # Maintient la taille maximale
        while sum(len(item.content) for item in self.short_term) > self.max_context_length:
            self._compress_oldest_memory()
        
        # Met à jour le graphe si pertinent
        if metadata and "user_id" in metadata:
            self._update_graph_from_conversation(
                items,
                user_id=metadata["user_id"]
            )

    def retrieve_relevant_context(
        self,
        current_input: str,
        max_turns: int = 10,
        min_similarity: float = 0.5
    ) -> List[str]:
        """Récupère le contexte pertinent.
        
        Args:
            current_input: Message courant
            max_turns: Nombre max de tours à retourner
            min_similarity: Similarité minimale

        Returns:
            Liste des messages contextuels pertinents
        """
        # Calcule l'embedding de la requête
        query_embedding = self.embedding_model.encode(
            current_input,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Cherche dans toutes les mémoires
        all_memories = (
            self.short_term +
            self.long_term +
            self.compressed
        )
        
        # Calcule les similarités
        similarities = []
        for item in all_memories:
            sim = F.cosine_similarity(
                query_embedding,
                torch.from_numpy(item.embedding.vector),
                dim=0
            ).item()
            similarities.append((sim, item))
            
        # Trie et filtre
        relevant = sorted(
            [(s, m) for s, m in similarities if s >= min_similarity],
            key=lambda x: x[0],
            reverse=True
        )
        
        return [m.content for _, m in relevant[:max_turns]]

    def compress_old_conversations(
        self,
        compression_threshold: int = 100
    ) -> None:
        """Compresse les anciennes conversations.
        
        Args:
            compression_threshold: Seuil de compression
        """
        # Identifie les items à compresser
        to_compress = []
        
        for memory_list in [self.short_term, self.long_term]:
            compress_candidates = [
                item for item in memory_list
                if item.should_compress(threshold=0.3)
            ]
            
            if len(compress_candidates) >= compression_threshold:
                to_compress.extend(compress_candidates)
                memory_list = [
                    item for item in memory_list
                    if item not in compress_candidates
                ]
        
        if not to_compress:
            return
            
        # Compresse par lots
        batch_size = 32
        for i in range(0, len(to_compress), batch_size):
            batch = to_compress[i:i + batch_size]
            
            # Crée un résumé
            embeddings = torch.stack([
                torch.from_numpy(item.embedding.vector)
                for item in batch
            ])
            
            mean_embedding = scatter_mean(
                embeddings,
                torch.zeros(len(batch), dtype=torch.long),
                dim=0
            )[0]
            
            # Crée l'item compressé
            compressed_item = ConversationMemoryItem(
                content=f"Mémoire compressée ({len(batch)} items)",
                embedding=MemoryEmbedding(
                    vector=mean_embedding.cpu().numpy(),
                    metadata={"original_items": len(batch)}
                ),
                memory_type=MemoryType.COMPRESSED,
                importance_score=np.mean([
                    item.importance_score for item in batch
                ])
            )
            
            self.compressed.append(compressed_item)
            
        logger.info(
            f"Compression: {len(to_compress)} items -> "
            f"{len(self.compressed)} résumés"
        )

    def build_user_profile(
        self,
        conversation_history: List[Dict[str, str]],
        user_id: str
    ) -> UserProfile:
        """Construit un profil utilisateur.
        
        Args:
            conversation_history: Historique des conversations
            user_id: ID de l'utilisateur

        Returns:
            Profil utilisateur
        """
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
        else:
            profile = UserProfile(user_id=user_id)
            self.user_profiles[user_id] = profile
            
        # Analyse chaque message
        for msg in conversation_history:
            if msg.get("role") == "user":
                # Crée l'embedding
                content = msg["content"]
                embedding = self.embedding_model.encode(
                    content,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Met à jour le profil
                profile.update_from_conversation(
                    content,
                    embedding.cpu().numpy()
                )
                
        return profile

    def update_relationship_graph(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> None:
        """Met à jour le graphe de relations.
        
        Args:
            entities: Liste des entités à ajouter/mettre à jour
            relationships: Liste des relations à ajouter/mettre à jour
        """
        # Ajoute/met à jour les entités
        for entity_data in entities:
            entity = KnowledgeGraphEntity(
                id=entity_data["id"],
                type=EntityType(entity_data["type"]),
                name=entity_data["name"],
                attributes=entity_data.get("attributes", {})
            )
            
            # Calcule l'embedding si nécessaire
            if "embedding" not in entity_data and entity.name:
                embedding = self.embedding_model.encode(
                    entity.name,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                entity.embedding = embedding.cpu().numpy()
                
            self.graph.add_entity(entity)
            
        # Ajoute les relations
        for rel_data in relationships:
            relation = KnowledgeGraphRelation(
                source_id=rel_data["source"],
                target_id=rel_data["target"],
                type=RelationType(rel_data["type"]),
                weight=rel_data.get("weight", 1.0),
                attributes=rel_data.get("attributes", {})
            )
            self.graph.add_relation(relation)
            
        # Mise à jour du modèle GNN si nécessaire
        if entities or relationships:
            self._update_gnn()

    def _estimate_importance(
        self,
        text: str,
        embedding: torch.Tensor
    ) -> float:
        """Estime l'importance d'un message."""
        # TODO: Utiliser un modèle plus sophistiqué
        # Pour l'instant: heuristique simple basée sur la longueur
        return min(len(text.split()) / 100, 1.0)

    def _compress_oldest_memory(self) -> None:
        """Compresse le plus ancien souvenir."""
        if not self.short_term:
            return
            
        # Trouve le plus ancien
        oldest = min(
            self.short_term,
            key=lambda x: x.embedding.timestamp
        )
        
        # Déplace vers la mémoire long terme
        self.short_term.remove(oldest)
        oldest.memory_type = MemoryType.LONG_TERM
        self.long_term.append(oldest)
        
        # Compresse si nécessaire
        if len(self.long_term) > self.max_context_length:
            self.compress_old_conversations()

    def _update_graph_from_conversation(
        self,
        items: List[ConversationMemoryItem],
        user_id: str
    ) -> None:
        """Met à jour le graphe depuis une conversation."""
        # TODO: Extraire les entités et relations
        pass

    def _update_gnn(self) -> None:
        """Met à jour les embeddings du graphe via le GNN."""
        # Convertit en format PyTorch Geometric
        graph_data = self.graph.to_pytorch_geometric()
        graph_data = graph_data.to(self.device)
        
        # Passe dans le GNN
        self.gnn.train()
        with torch.no_grad():
            node_embeddings = self.gnn(graph_data)
            
        # Met à jour les embeddings des entités
        for i, entity_id in enumerate(self.graph.entities):
            entity = self.graph.entities[entity_id]
            entity.embedding = node_embeddings[i].cpu().numpy()