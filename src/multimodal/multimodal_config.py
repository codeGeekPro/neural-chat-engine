"""Configuration des capacités multimodales."""

import os
from typing import Dict, Any, Optional
from pathlib import Path


class MultimodalConfig:
    """Configuration centralisée pour les composants multimodaux."""

    def __init__(self):
        self.config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Charge la configuration par défaut."""
        return {
            # Configuration Vision
            "vision": {
                "enabled": True,
                "model_name": "openai/clip-vit-base-patch32",
                "description_model": "Salesforce/blip-image-captioning-base",
                "detection_model": "facebook/detr-resnet-50",
                "ocr_enabled": True,
                "max_objects": 10,
                "confidence_threshold": 0.5,
                "detail_level": "detailed",
                "cache_embeddings": True,
                "device": "auto",  # auto, cpu, cuda
                "batch_size": 1,
                "max_image_size": 1024,
                "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            },

            # Configuration Audio
            "audio": {
                "enabled": True,
                "whisper_model": "openai/whisper-base",
                "wav2vec_model": "facebook/wav2vec2-base-960h",
                "tts_model": "microsoft/speecht5_tts",
                "sample_rate": 16000,
                "language": "fr",
                "voice_gender": "neutral",
                "speaking_rate": 1.0,
                "device": "auto",
                "cache_audio": True,
                "max_audio_length": 300,  # secondes
                "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
                "diarization_threshold": 0.7,
                "embedding_dim": 512
            },

            # Configuration Multimodale
            "multimodal": {
                "fusion_enabled": True,
                "cross_modal_attention": True,
                "embedding_fusion": "concat",  # concat, mean, attention
                "max_sequence_length": 512,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9
            },

            # Configuration Performance
            "performance": {
                "gpu_memory_fraction": 0.8,
                "cpu_threads": None,  # auto
                "preload_models": False,
                "model_cache_dir": "./models",
                "temp_dir": "./temp",
                "cleanup_temp_files": True,
                "max_concurrent_requests": 4
            },

            # Configuration Sécurité
            "security": {
                "max_file_size": 50 * 1024 * 1024,  # 50MB
                "allowed_extensions": [".jpg", ".png", ".wav", ".mp3", ".txt"],
                "scan_for_viruses": False,
                "validate_inputs": True,
                "timeout_seconds": 300
            },

            # Configuration Logging
            "logging": {
                "level": "INFO",
                "file_path": "logs/multimodal.log",
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5,
                "console_output": True
            }
        }

    def get_vision_config(self) -> Dict[str, Any]:
        """Retourne la configuration vision."""
        return self.config["vision"].copy()

    def get_audio_config(self) -> Dict[str, Any]:
        """Retourne la configuration audio."""
        return self.config["audio"].copy()

    def get_multimodal_config(self) -> Dict[str, Any]:
        """Retourne la configuration multimodale."""
        return self.config["multimodal"].copy()

    def get_performance_config(self) -> Dict[str, Any]:
        """Retourne la configuration performance."""
        return self.config["performance"].copy()

    def get_security_config(self) -> Dict[str, Any]:
        """Retourne la configuration sécurité."""
        return self.config["security"].copy()

    def get_logging_config(self) -> Dict[str, Any]:
        """Retourne la configuration logging."""
        return self.config["logging"].copy()

    def update_config(self, section: str, key: str, value: Any):
        """Met à jour une valeur de configuration."""
        if section not in self.config:
            raise ValueError(f"Section '{section}' inconnue")
        if key not in self.config[section]:
            raise ValueError(f"Clé '{key}' inconnue dans la section '{section}'")

        self.config[section][key] = value

    def load_from_file(self, config_path: str):
        """Charge la configuration depuis un fichier JSON."""
        import json

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)

            # Fusion récursive avec la config par défaut
            self._merge_configs(self.config, file_config)

        except FileNotFoundError:
            print(f"Fichier de configuration non trouvé: {config_path}")
        except json.JSONDecodeError as e:
            print(f"Erreur de parsing JSON: {e}")

    def save_to_file(self, config_path: str):
        """Sauvegarde la configuration vers un fichier JSON."""
        import json

        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Fusionne récursivement deux configurations."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value

    def validate_config(self) -> bool:
        """Valide la configuration actuelle."""
        try:
            # Validation des chemins
            if not os.path.isdir(self.config["performance"]["model_cache_dir"]):
                os.makedirs(self.config["performance"]["model_cache_dir"], exist_ok=True)

            if not os.path.isdir(self.config["performance"]["temp_dir"]):
                os.makedirs(self.config["performance"]["temp_dir"], exist_ok=True)

            # Validation des tailles de fichiers
            max_size = self.config["security"]["max_file_size"]
            if max_size <= 0:
                raise ValueError("max_file_size doit être positif")

            # Validation des seuils
            vision_threshold = self.config["vision"]["confidence_threshold"]
            if not 0 <= vision_threshold <= 1:
                raise ValueError("confidence_threshold doit être entre 0 et 1")

            audio_threshold = self.config["audio"]["diarization_threshold"]
            if not 0 <= audio_threshold <= 1:
                raise ValueError("diarization_threshold doit être entre 0 et 1")

            return True

        except Exception as e:
            print(f"Erreur de validation de configuration: {e}")
            return False

    def get_device(self, preferred_device: Optional[str] = None) -> str:
        """Détermine le device à utiliser (CPU/GPU)."""
        if preferred_device:
            return preferred_device

        device = self.config["vision"]["device"]
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

        return device

    def create_default_config_file(self, config_path: str = "config/multimodal_config.json"):
        """Crée un fichier de configuration par défaut."""
        self.save_to_file(config_path)
        print(f"Fichier de configuration créé: {config_path}")


# Instance globale de configuration
multimodal_config = MultimodalConfig()


def get_config() -> MultimodalConfig:
    """Retourne l'instance globale de configuration."""
    return multimodal_config


def load_config_from_file(config_path: str):
    """Charge la configuration depuis un fichier."""
    multimodal_config.load_from_file(config_path)


def save_config_to_file(config_path: str):
    """Sauvegarde la configuration vers un fichier."""
    multimodal_config.save_to_file(config_path)


# Configuration rapide pour développement
def create_dev_config():
    """Crée une configuration optimisée pour le développement."""
    config = MultimodalConfig()

    # Configuration légère pour le développement
    config.update_config("vision", "model_name", "openai/clip-vit-base-patch16")  # Modèle plus petit
    config.update_config("audio", "whisper_model", "openai/whisper-tiny")  # Modèle tiny
    config.update_config("performance", "preload_models", False)
    config.update_config("performance", "max_concurrent_requests", 1)
    config.update_config("logging", "level", "DEBUG")

    return config


# Configuration de production
def create_prod_config():
    """Crée une configuration optimisée pour la production."""
    config = MultimodalConfig()

    # Configuration optimisée pour la production
    config.update_config("vision", "model_name", "openai/clip-vit-large-patch14")  # Modèle large
    config.update_config("audio", "whisper_model", "openai/whisper-large-v3")  # Modèle large
    config.update_config("performance", "preload_models", True)
    config.update_config("performance", "gpu_memory_fraction", 0.9)
    config.update_config("performance", "max_concurrent_requests", 8)
    config.update_config("logging", "level", "WARNING")

    return config


if __name__ == "__main__":
    # Test de la configuration
    config = MultimodalConfig()

    print("Configuration Vision:")
    vision_config = config.get_vision_config()
    for key, value in vision_config.items():
        print(f"  {key}: {value}")

    print("\nConfiguration Audio:")
    audio_config = config.get_audio_config()
    for key, value in audio_config.items():
        print(f"  {key}: {value}")

    # Validation
    if config.validate_config():
        print("\n✅ Configuration valide")
    else:
        print("\n❌ Configuration invalide")

    # Création du fichier de configuration par défaut
    config.create_default_config_file()