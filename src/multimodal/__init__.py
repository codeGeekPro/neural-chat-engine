"""Module de capacités multimodales."""

from .vision_processor import VisionProcessor
from .audio_processor import AudioProcessor
from .multimodal_types import (
    ImageAnalysis,
    AudioAnalysis,
    VisualQuestion,
    SpeechSynthesis,
    MultimodalConfig
)
from .multimodal_config import (
    MultimodalConfig as ConfigManager,
    get_config,
    load_config_from_file,
    save_config_to_file,
    create_dev_config,
    create_prod_config
)

__all__ = [
    "VisionProcessor",
    "AudioProcessor",
    "ImageAnalysis",
    "AudioAnalysis",
    "VisualQuestion",
    "SpeechSynthesis",
    "MultimodalConfig",
    "ConfigManager",
    "get_config",
    "load_config_from_file",
    "save_config_to_file",
    "create_dev_config",
    "create_prod_config"
]