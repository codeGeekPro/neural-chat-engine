from __future__ import annotations

"""Processeur de conversations multilingue avec métriques et sérialisation."""

import csv
import html
import io
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from .conversation_types import ConversationFormat, ConversationMetrics, ConversationStats


class ConversationProcessor:
    """Nettoyage et normalisation de texte multilingue pour conversations."""

    URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    INLINE_CODE_RE = re.compile(r"`[^`]+`")
    MULTISPACE_RE = re.compile(r"\s{2,}")

    def __init__(self) -> None:
        self.metrics = ConversationMetrics()
        self._reset_session()

    def _reset_session(self) -> None:
        """Réinitialise la session courante."""
        self.start_time = None
        self.current_stats = ConversationStats()
    
    def start_session(self) -> None:
        """Démarre une nouvelle session de conversation."""
        self._reset_session()
        self.start_time = datetime.now()
        self.current_stats.start_time = self.start_time

    def clean_text(self, text: str) -> str:
        """Nettoie et normalise un texte."""
        if not text:
            return ""
            
        # Préserve les blocs de code
        code_blocks = {}
        for i, m in enumerate(self.CODE_BLOCK_RE.finditer(text)):
            key = f"__CODE_BLOCK_{i}__"
            code_blocks[key] = m.group()
            text = text.replace(m.group(), key)
            
        # Préserve les codes en ligne
        inline_codes = {}
        for i, m in enumerate(self.INLINE_CODE_RE.finditer(text)):
            key = f"__INLINE_CODE_{i}__"
            inline_codes[key] = m.group()
            text = text.replace(m.group(), key)
            
        # Nettoyage du texte
        text = text.strip()
        text = self.URL_RE.sub("[URL]", text)
        text = self.MULTISPACE_RE.sub(" ", text)
        
        # Restaure les blocs de code
        for key, code in code_blocks.items():
            text = text.replace(key, code)
        for key, code in inline_codes.items():
            text = text.replace(key, code)
            
        return text

    def normalize_conversation(self, convo: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Normalise une conversation [(role, content), ...]."""
        norm = []
        for role, content in convo:
            role = role.lower().strip()
            if role not in {"user", "assistant", "system"}:
                role = "user"  # fallback
            norm.append((role, self.clean_text(content)))
        return norm

    def normalize_many(self, convos: Iterable[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        """Normalise plusieurs conversations."""
        return [self.normalize_conversation(c) for c in convos]

    def save_conversation(self, convo: List[Tuple[str, str]], fmt: ConversationFormat) -> str:
        """Sérialise une conversation dans le format spécifié."""
        normalized = self.normalize_conversation(convo)
        
        if fmt == ConversationFormat.JSON:
            return self._to_json(normalized)
        elif fmt == ConversationFormat.JSONL:
            return self._to_jsonl(normalized)
        elif fmt == ConversationFormat.CSV:
            return self._to_csv(normalized)
        elif fmt == ConversationFormat.XML:
            return self._to_xml(normalized)
        elif fmt == ConversationFormat.MARKDOWN:
            return self._to_markdown(normalized)
        elif fmt == ConversationFormat.TEXT:
            return self._to_text(normalized)
        else:
            raise ValueError(f"Format non supporté: {fmt}")

    def load_conversation(self, data: str, fmt: ConversationFormat) -> List[Tuple[str, str]]:
        """Charge une conversation depuis le format spécifié."""
        if fmt == ConversationFormat.JSON:
            return self._from_json(data)
        elif fmt == ConversationFormat.JSONL:
            return self._from_jsonl(data)
        elif fmt == ConversationFormat.CSV:
            return self._from_csv(data)
        elif fmt == ConversationFormat.XML:
            return self._from_xml(data)
        elif fmt == ConversationFormat.MARKDOWN:
            return self._from_markdown(data)
        elif fmt == ConversationFormat.TEXT:
            return self._from_text(data)
        else:
            raise ValueError(f"Format non supporté: {fmt}")

    def analyze_conversation(self, convo: List[Tuple[str, str]]) -> ConversationStats:
        """Analyse une conversation et calcule ses statistiques."""
        # S'assurer que la session est démarrée
        if self.start_time is None:
            self.start_session()

        # Calculer les métriques basiques
        self.current_stats.num_turns = len(convo)
        self.current_stats.num_tokens = sum(len(content.split()) for _, content in convo)

        # Calculer la durée
        self.current_stats.end_time = datetime.now()
        self.current_stats.duration_seconds = (
            self.current_stats.end_time - self.start_time
        ).total_seconds()

        # Calculer les métriques de qualité via ConversationMetrics
        self.current_stats.coherence_score = self.metrics._estimate_coherence(convo)
        self.current_stats.engagement_score = self.metrics._estimate_engagement(convo)

        return self.current_stats

    def _to_json(self, convo: List[Tuple[str, str]]) -> str:
        """Convertit en JSON."""
        data = [{"role": role, "content": content} for role, content in convo]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _from_json(self, data: str) -> List[Tuple[str, str]]:
        """Charge depuis JSON."""
        messages = json.loads(data)
        return [(m["role"], m["content"]) for m in messages]

    def _to_jsonl(self, convo: List[Tuple[str, str]]) -> str:
        """Convertit en JSONL."""
        lines = []
        for role, content in convo:
            line = json.dumps({"role": role, "content": content}, ensure_ascii=False)
            lines.append(line)
        return "\n".join(lines)

    def _from_jsonl(self, data: str) -> List[Tuple[str, str]]:
        """Charge depuis JSONL."""
        convo = []
        for line in data.strip().split("\n"):
            if line:
                msg = json.loads(line)
                convo.append((msg["role"], msg["content"]))
        return convo

    def _to_csv(self, convo: List[Tuple[str, str]]) -> str:
        """Convertit en CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["role", "content"])  # header
        writer.writerows(convo)
        return output.getvalue()

    def _from_csv(self, data: str) -> List[Tuple[str, str]]:
        """Charge depuis CSV."""
        convo = []
        reader = csv.reader(io.StringIO(data))
        next(reader)  # skip header
        for row in reader:
            if len(row) == 2:
                convo.append((row[0], row[1]))
        return convo

    def _to_xml(self, convo: List[Tuple[str, str]]) -> str:
        """Convertit en XML."""
        root = ET.Element("conversation")
        for role, content in convo:
            msg = ET.SubElement(root, "message")
            role_elem = ET.SubElement(msg, "role")
            role_elem.text = role
            content_elem = ET.SubElement(msg, "content")
            content_elem.text = content
        return ET.tostring(root, encoding="unicode", method="xml")

    def _from_xml(self, data: str) -> List[Tuple[str, str]]:
        """Charge depuis XML."""
        convo = []
        root = ET.fromstring(data)
        for msg in root.findall("message"):
            role = msg.find("role")
            content = msg.find("content")
            if role is not None and content is not None:
                convo.append((role.text or "", content.text or ""))
        return convo

    def _to_markdown(self, convo: List[Tuple[str, str]]) -> str:
        """Convertit en Markdown."""
        lines = []
        for role, content in convo:
            lines.append(f"## {role.title()}")
            lines.append(content)
            lines.append("")  # ligne vide pour séparation
        return "\n".join(lines)

    def _from_markdown(self, data: str) -> List[Tuple[str, str]]:
        """Charge depuis Markdown."""
        convo = []
        current_role = None
        current_content = []
        
        for line in data.split("\n"):
            line = line.strip()
            if line.startswith("## "):
                if current_role and current_content:
                    convo.append((current_role, "\n".join(current_content).strip()))
                    current_content = []
                current_role = line[3:].lower()
            elif current_role and line:
                current_content.append(line)
                
        if current_role and current_content:
            convo.append((current_role, "\n".join(current_content).strip()))
            
        return convo

    def _to_text(self, convo: List[Tuple[str, str]]) -> str:
        """Convertit en texte brut."""
        lines = []
        for role, content in convo:
            lines.append(f"{role}:")
            lines.append(content)
            lines.append("")  # ligne vide pour séparation
        return "\n".join(lines)

    def _from_text(self, data: str) -> List[Tuple[str, str]]:
        """Charge depuis texte brut."""
        convo = []
        lines = data.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.endswith(":"):
                role = line[:-1].strip()
                content_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    content_lines.append(lines[i].strip())
                    i += 1
                if content_lines:
                    convo.append((role, "\n".join(content_lines)))
            i += 1
        return convo



    def normalize_many(self, convos: Iterable[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        return [self.normalize_conversation(c) for c in convos]
