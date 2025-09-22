from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from .conversation_processor import ConversationProcessor
from .data_augmentation import DataAugmentation
from .embedding_generator import EmbeddingGenerator


Conversation = List[Tuple[str, str]]  # [(role, content), ...]


@dataclass
class DataPipeline:
    output_dir: str = "data/processed"
    languages: List[str] = field(default_factory=lambda: ["fr", "en"])
    max_items: int = 1000

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        self.processor = ConversationProcessor()
        self.augmenter = DataAugmentation()
        self.embedder = EmbeddingGenerator()

    # ============ EXTRACT ============
    def extract_conversations_from_github(self, repo_urls: Iterable[str]) -> List[Conversation]:
        """Extract conversations from GitHub issues/PRs comments.

        Placeholder implementation: returns small synthetic samples.
        In production, use GitHub API (GraphQL/REST) with pagination and auth.
        """
        convos: List[Conversation] = []
        for url in repo_urls:
            # synth sample from repo URL
            convos.append([
                ("user", f"Bonjour, j'ai un problème avec le repo {url}"),
                ("assistant", "Pouvez-vous fournir les logs et la version ?"),
            ])
            if len(convos) >= self.max_items:
                break
        return convos

    def extract_from_stackoverflow(self, tags: Iterable[str]) -> List[Conversation]:
        """Extract Q/A from Stack Overflow.

        Placeholder implementation: returns synthetic Q/A pairs.
        In production, use Stack Exchange API with tag filters and backoff.
        """
        convos: List[Conversation] = []
        for t in tags:
            convos.append([
                ("user", f"How to fix import error with pydantic on tag {t}?"),
                ("assistant", "Install pydantic-settings and update imports for V2."),
            ])
            if len(convos) >= self.max_items:
                break
        return convos

    # ============ TRANSFORM ============
    def clean_and_normalize_text(self, convos: List[Conversation]) -> List[Conversation]:
        return self.processor.normalize_many(convos)

    def augment_conversation_data(self, convos: List[Conversation]) -> List[Conversation]:
        return [self.augmenter.augment_conversation(c) for c in convos]

    def create_sentence_embeddings(self, convos: List[Conversation]) -> List[List[float]]:
        # Simple approach: concatenate last user + assistant message per convo
        texts: List[str] = []
        for conv in convos:
            snippet = " ".join(content for _, content in conv[-2:]) if len(conv) >= 2 else " ".join(c for _, c in conv)
            texts.append(snippet)
        return self.embedder.encode(texts)

    # ============ LOAD ============
    def save_processed_data(self, convos: List[Conversation], embeddings: List[List[float]]) -> None:
        jsonl_path = os.path.join(self.output_dir, "conversations.jsonl")
        csv_path = os.path.join(self.output_dir, "embeddings.csv")

        # Save conversations as JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for conv in convos:
                f.write(json.dumps({"conversation": conv}, ensure_ascii=False) + "\n")

        # Save embeddings as CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for vec in embeddings:
                writer.writerow([f"{v:.6f}" for v in vec])

    # ============ ORCHESTRATION ============
    def run(self, repo_urls: Iterable[str], tags: Iterable[str]) -> None:
        # Extract
        gh = self.extract_conversations_from_github(repo_urls)
        so = self.extract_from_stackoverflow(tags)
        all_convos = (gh + so)[: self.max_items]

        # Transform
        all_convos = self.clean_and_normalize_text(all_convos)
        all_convos = self.augment_conversation_data(all_convos)
        embs = self.create_sentence_embeddings(all_convos)

        # Load
        self.save_processed_data(all_convos, embs)
