"""Types et structures pour les capacités multimodales."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np


class VisionModelType(Enum):
    """Types de modèles de vision."""
    CLIP = "clip"
    BLIP = "blip"
    DETR = "detr"
    OWL_VIT = "owl_vit"


class AudioModelType(Enum):
    """Types de modèles audio."""
    WHISPER = "whisper"
    WAV2VEC2 = "wav2vec2"
    SPEECH_BRAIN = "speech_brain"


class DetailLevel(Enum):
    """Niveaux de détail pour les descriptions."""
    BRIEF = "brief"
    MEDIUM = "medium"
    DETAILED = "detailed"


class VoiceGender(Enum):
    """Genres de voix pour la synthèse."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class Language(Enum):
    """Langues supportées."""
    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"


@dataclass
class MultimodalConfig:
    """Configuration pour les capacités multimodales."""

    vision_model: VisionModelType = VisionModelType.CLIP
    audio_model: AudioModelType = AudioModelType.WHISPER
    device: str = "cpu"
    cache_dir: Optional[Path] = None
    max_image_size: Tuple[int, int] = (224, 224)
    supported_languages: List[Language] = field(default_factory=lambda: [Language.ENGLISH, Language.FRENCH])
    enable_ocr: bool = True
    enable_object_detection: bool = True
    enable_scene_description: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "vision_model": self.vision_model.value,
            "audio_model": self.audio_model.value,
            "device": self.device,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "max_image_size": self.max_image_size,
            "supported_languages": [lang.value for lang in self.supported_languages],
            "enable_ocr": self.enable_ocr,
            "enable_object_detection": self.enable_object_detection,
            "enable_scene_description": self.enable_scene_description
        }


@dataclass
class ImageAnalysis:
    """Résultat d'analyse d'image."""

    image_path: Path
    description: str
    objects_detected: List[Dict[str, Any]] = field(default_factory=list)
    scene_description: str = ""
    dominant_colors: List[Tuple[str, float]] = field(default_factory=list)
    text_extracted: str = ""
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "image_path": str(self.image_path),
            "description": self.description,
            "objects_detected": self.objects_detected,
            "scene_description": self.scene_description,
            "dominant_colors": self.dominant_colors,
            "text_extracted": self.text_extracted,
            "confidence_scores": self.confidence_scores,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class VisualQuestion:
    """Question visuelle avec réponse."""

    question: str
    image_path: Path
    answer: str
    confidence: float
    reasoning: str = ""
    relevant_objects: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "question": self.question,
            "image_path": str(self.image_path),
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "relevant_objects": self.relevant_objects,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AudioAnalysis:
    """Résultat d'analyse audio."""

    audio_path: Path
    transcription: str
    language: Language
    speaker_count: int = 1
    speakers_identified: List[Dict[str, Any]] = field(default_factory=list)
    duration: float = 0.0
    confidence: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "audio_path": str(self.audio_path),
            "transcription": self.transcription,
            "language": self.language.value,
            "speaker_count": self.speaker_count,
            "speakers_identified": self.speakers_identified,
            "duration": self.duration,
            "confidence": self.confidence,
            "segments": self.segments,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SpeechSynthesis:
    """Configuration pour la synthèse vocale."""

    text: str
    voice_gender: VoiceGender = VoiceGender.NEUTRAL
    language: Language = Language.ENGLISH
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    voice_name: Optional[str] = None
    custom_voice_path: Optional[Path] = None
    output_path: Optional[Path] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "text": self.text,
            "voice_gender": self.voice_gender.value,
            "language": self.language.value,
            "speaking_rate": self.speaking_rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "voice_name": self.voice_name,
            "custom_voice_path": str(self.custom_voice_path) if self.custom_voice_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OCRResult:
    """Résultat d'extraction de texte depuis une image."""

    text: str
    confidence: float
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    language: Language = Language.ENGLISH
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_boxes": self.bounding_boxes,
            "language": self.language.value,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ObjectDetection:
    """Résultat de détection d'objets."""

    objects: List[Dict[str, Any]] = field(default_factory=list)
    confidence_threshold: float = 0.5
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "objects": self.objects,
            "confidence_threshold": self.confidence_threshold,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }

    def get_top_objects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retourne les objets les plus confiants."""
        sorted_objects = sorted(
            self.objects,
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )
        return sorted_objects[:limit]


@dataclass
class SceneDescription:
    """Description de scène."""

    description: str
    tags: List[str] = field(default_factory=list)
    mood: str = ""
    lighting: str = ""
    composition: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "description": self.description,
            "tags": self.tags,
            "mood": self.mood,
            "lighting": self.lighting,
            "composition": self.composition,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SpeakerIdentification:
    """Résultat d'identification de locuteurs."""

    speakers: List[Dict[str, Any]] = field(default_factory=list)
    speaker_count: int = 0
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "speakers": self.speakers,
            "speaker_count": self.speaker_count,
            "confidence_scores": self.confidence_scores,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


# Types pour les embeddings et représentations
ImageEmbedding = np.ndarray
AudioEmbedding = np.ndarray
TextEmbedding = np.ndarray

# Types pour les chemins et données
ImageInput = Union[str, Path, np.ndarray, Any]  # PIL Image, etc.
AudioInput = Union[str, Path, np.ndarray, Any]  # Audio array, etc.