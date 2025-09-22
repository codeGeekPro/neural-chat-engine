"""
Configuration Loader - Neural Chat Engine

Utilitaires pour charger et valider les configurations YAML,
gérer les profils d'environnement, et merger les configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass, field

from .config import Settings, get_settings


logger = logging.getLogger(__name__)


@dataclass
class ConfigProfile:
    """Profil de configuration pour différents environnements."""
    name: str
    description: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    enabled_features: List[str] = field(default_factory=list)
    disabled_features: List[str] = field(default_factory=list)


class ConfigurationLoader:
    """Chargeur de configuration avancé pour le chatbot."""
    
    def __init__(self, base_settings: Optional[Settings] = None):
        """
        Initialise le chargeur de configuration.
        
        Args:
            base_settings: Settings de base, utilise get_settings() par défaut
        """
        self.settings = base_settings or get_settings()
        self.configs: Dict[str, Any] = {}
        self.profiles: Dict[str, ConfigProfile] = {}
        self._load_default_profiles()
    
    def _load_default_profiles(self) -> None:
        """Charge les profils de configuration par défaut."""
        self.profiles = {
            "development": ConfigProfile(
                name="development",
                description="Configuration pour développement local",
                config_overrides={
                    "logging.log_level": "DEBUG",
                    "server.debug": True,
                    "models.enable_model_caching": False,
                    "analytics.track_user_satisfaction": False
                },
                enabled_features=["debug_mode", "hot_reload", "mock_apis"]
            ),
            
            "staging": ConfigProfile(
                name="staging",
                description="Configuration pour environnement de test",
                config_overrides={
                    "logging.log_level": "INFO",
                    "server.debug": False,
                    "analytics.track_user_satisfaction": True,
                    "safety.data_retention.conversation_logs_days": 30
                },
                enabled_features=["analytics", "monitoring"],
                disabled_features=["debug_mode"]
            ),
            
            "production": ConfigProfile(
                name="production", 
                description="Configuration pour production",
                config_overrides={
                    "logging.log_level": "WARNING",
                    "server.debug": False,
                    "safety.encrypt_sensitive_data": True,
                    "analytics.track_user_satisfaction": True,
                    "performance_thresholds.intent_classification.min_confidence": 0.8
                },
                enabled_features=["analytics", "monitoring", "security", "compliance"],
                disabled_features=["debug_mode", "mock_apis"]
            )
        }
    
    def load_yaml_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Charge un fichier de configuration YAML.
        
        Args:
            config_path: Chemin vers le fichier YAML
            
        Returns:
            Configuration chargée
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            yaml.YAMLError: Si le YAML est invalide
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config_data
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration {config_path}: {e}")
            raise
    
    def load_main_config(self) -> Dict[str, Any]:
        """Charge la configuration principale du chatbot."""
        config_path = Path("configs/chatbot_config.yaml")
        
        if not config_path.exists():
            logger.warning(f"Main config file not found: {config_path}")
            return self._get_default_config()
        
        config = self.load_yaml_config(config_path)
        self.configs["main"] = config
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Retourne une configuration par défaut minimale."""
        return {
            "intent_categories": {
                "general": {
                    "name": "General Conversation",
                    "intents": ["greeting", "goodbye", "thank_you"]
                }
            },
            "personality_profiles": {
                "default": {
                    "name": "Default Assistant",
                    "traits": {
                        "formality": 0.5,
                        "empathy": 0.7,
                        "technical_depth": 0.6
                    }
                }
            },
            "languages": {
                "en": {"name": "English", "code": "en-US"},
                "fr": {"name": "Français", "code": "fr-FR"}
            }
        }
    
    def apply_profile(self, profile_name: str) -> None:
        """
        Applique un profil de configuration.
        
        Args:
            profile_name: Nom du profil à appliquer
            
        Raises:
            ValueError: Si le profil n'existe pas
        """
        if profile_name not in self.profiles:
            raise ValueError(f"Profile '{profile_name}' not found. Available: {list(self.profiles.keys())}")
        
        profile = self.profiles[profile_name]
        logger.info(f"Applying configuration profile: {profile.name}")
        
        # Application des overrides
        for key, value in profile.config_overrides.items():
            self._set_nested_value(self.configs, key, value)
        
        # Gestion des features
        self._apply_feature_flags(profile.enabled_features, profile.disabled_features)
        
        logger.info(f"Profile '{profile_name}' applied successfully")
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        Définit une valeur dans une configuration imbriquée.
        
        Args:
            config: Configuration à modifier
            key_path: Chemin de la clé (ex: "logging.log_level")
            value: Valeur à définir
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _apply_feature_flags(self, enabled: List[str], disabled: List[str]) -> None:
        """
        Applique les flags de fonctionnalités.
        
        Args:
            enabled: Fonctionnalités à activer
            disabled: Fonctionnalités à désactiver
        """
        # Logique d'application des feature flags
        # À implémenter selon les besoins spécifiques
        logger.debug(f"Enabled features: {enabled}")
        logger.debug(f"Disabled features: {disabled}")
    
    def get_intent_categories(self) -> Dict[str, Any]:
        """Retourne les catégories d'intentions configurées."""
        main_config = self.configs.get("main", {})
        return main_config.get("intent_categories", {})
    
    def get_personality_profiles(self) -> Dict[str, Any]:
        """Retourne les profils de personnalité configurés."""
        main_config = self.configs.get("main", {})
        return main_config.get("personality_profiles", {})
    
    def get_response_templates(self) -> Dict[str, Any]:
        """Retourne les templates de réponse configurés."""
        main_config = self.configs.get("main", {})
        return main_config.get("response_templates", {})
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Retourne les langues supportées."""
        main_config = self.configs.get("main", {})
        return main_config.get("languages", {})
    
    def get_conversation_rules(self) -> Dict[str, Any]:
        """Retourne les règles de conversation."""
        main_config = self.configs.get("main", {})
        return main_config.get("conversation_rules", {})
    
    def get_performance_thresholds(self) -> Dict[str, Any]:
        """Retourne les seuils de performance."""
        main_config = self.configs.get("main", {})
        return main_config.get("performance_thresholds", {})
    
    def get_safety_config(self) -> Dict[str, Any]:
        """Retourne la configuration de sécurité."""
        main_config = self.configs.get("main", {})
        return main_config.get("safety", {})
    
    def validate_configuration(self) -> bool:
        """
        Valide la cohérence de la configuration complète.
        
        Returns:
            True si la configuration est valide
        """
        is_valid = True
        
        try:
            # Validation des catégories d'intentions
            intent_categories = self.get_intent_categories()
            if not intent_categories:
                logger.error("No intent categories configured")
                is_valid = False
            
            # Validation des profils de personnalité
            personality_profiles = self.get_personality_profiles()
            if not personality_profiles:
                logger.error("No personality profiles configured")
                is_valid = False
            
            # Validation des langues supportées
            languages = self.get_supported_languages()
            if not languages:
                logger.error("No supported languages configured")
                is_valid = False
            
            # Validation des seuils de performance
            thresholds = self.get_performance_thresholds()
            if thresholds:
                intent_thresholds = thresholds.get("intent_classification", {})
                min_confidence = intent_thresholds.get("min_confidence", 0)
                if min_confidence < 0.5 or min_confidence > 1.0:
                    logger.warning(f"Intent confidence threshold should be between 0.5 and 1.0, got {min_confidence}")
            
            logger.info(f"Configuration validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            is_valid = False
        
        return is_valid
    
    def export_config(self, output_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Exporte la configuration actuelle.
        
        Args:
            output_path: Chemin de sortie
            format: Format d'export ("yaml" ou "json")
        """
        output_path = Path(output_path)
        
        try:
            if format.lower() == "yaml":
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.configs, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.configs, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Configuration exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            raise
    
    def reload_configuration(self) -> None:
        """Recharge toute la configuration depuis les fichiers."""
        logger.info("Reloading configuration...")
        self.configs.clear()
        self.load_main_config()
        
        # Application du profil basé sur l'environnement
        env_profile = self.settings.environment.value
        if env_profile in self.profiles:
            self.apply_profile(env_profile)
        
        logger.info("Configuration reloaded successfully")


# Instance globale du chargeur de configuration
_config_loader: Optional[ConfigurationLoader] = None


def get_config_loader() -> ConfigurationLoader:
    """Factory function pour obtenir le chargeur de configuration."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader()
        _config_loader.load_main_config()
        
        # Application automatique du profil basé sur l'environnement
        settings = get_settings()
        env_profile = settings.environment.value
        if env_profile in _config_loader.profiles:
            _config_loader.apply_profile(env_profile)
    
    return _config_loader


def reload_config() -> None:
    """Recharge la configuration globale."""
    global _config_loader
    if _config_loader:
        _config_loader.reload_configuration()


if __name__ == "__main__":
    # Test du chargeur de configuration
    print("🔧 Neural Chat Engine - Configuration Loader Test")
    
    loader = get_config_loader()
    
    print(f"Intent categories: {len(loader.get_intent_categories())}")
    print(f"Personality profiles: {len(loader.get_personality_profiles())}")
    print(f"Supported languages: {len(loader.get_supported_languages())}")
    
    # Validation
    is_valid = loader.validate_configuration()
    print(f"Configuration valid: {is_valid}")
    
    # Test d'export
    # loader.export_config("config_export.yaml", "yaml")