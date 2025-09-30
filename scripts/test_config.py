#!/usr/bin/env python3
"""
Test Script - Configuration Management System
Script de test pour valider le système complet de gestion de configuration.
"""

import sys
import os
from pathlib import Path

# Ajout du path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.config import Settings, get_settings, setup_logging, validate_configuration
    from src.config_loader import get_config_loader, ConfigurationLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def test_basic_settings():
    """Test des settings de base."""
    print("🧪 Testing basic settings...")
    
    settings = get_settings()
    
    print(f"✅ Project name: {settings.project_name}")
    print(f"✅ Environment: {settings.environment.value}")
    print(f"✅ Debug mode: {settings.debug}")
    print(f"✅ Database URL: {settings.get_database_url()}")
    print(f"✅ Redis URL: {settings.redis.redis_url}")
    print(f"✅ Models directory: {settings.models_dir}")
    
    return True


def test_configuration_validation():
    """Test de validation de la configuration."""
    print("\n🧪 Testing configuration validation...")
    
    try:
        is_valid = validate_configuration()
        print(f"✅ Configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid
    except Exception as e:
        print(f"❌ Configuration validation error: {e}")
        return False


def test_yaml_config_loader():
    """Test du chargeur de configuration YAML."""
    print("\n🧪 Testing YAML config loader...")
    
    try:
        loader = get_config_loader()
        
        # Test des différentes sections
        intent_categories = loader.get_intent_categories()
        print(f"✅ Intent categories loaded: {len(intent_categories)} domains")
        
        personality_profiles = loader.get_personality_profiles()
        print(f"✅ Personality profiles loaded: {len(personality_profiles)} profiles")
        
        languages = loader.get_supported_languages()
        print(f"✅ Supported languages: {len(languages)} languages")
        
        templates = loader.get_response_templates()
        print(f"✅ Response templates loaded: {len(templates)} templates")
        
        # Validation
        is_valid = loader.validate_configuration()
        print(f"✅ YAML configuration validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return True
        
    except Exception as e:
        print(f"❌ YAML config loader error: {e}")
        return False


def test_environment_profiles():
    """Test des profils d'environnement."""
    print("\n🧪 Testing environment profiles...")
    
    try:
        loader = ConfigurationLoader()
        
        # Test des profils disponibles
        available_profiles = list(loader.profiles.keys())
        print(f"✅ Available profiles: {available_profiles}")
        
        # Test d'application d'un profil
        loader.apply_profile("development")
        print("✅ Development profile applied successfully")
        
        loader.apply_profile("production")
        print("✅ Production profile applied successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment profiles error: {e}")
        return False


def test_logging_setup():
    """Test de la configuration du logging."""
    print("\n🧪 Testing logging setup...")
    
    try:
        setup_logging()
        
        import logging
        logger = logging.getLogger("test_logger")
        
        logger.info("Test INFO message")
        logger.warning("Test WARNING message")
        logger.debug("Test DEBUG message")
        
        print("✅ Logging setup successful")
        return True
        
    except Exception as e:
        print(f"❌ Logging setup error: {e}")
        return False


def test_feature_flags():
    """Test des feature flags."""
    print("\n🧪 Testing feature flags...")
    
    try:
        settings = get_settings()
        
        feature_flags = {
            "conversation_memory": settings.enable_conversation_memory,
            "personality_adaptation": settings.enable_personality_adaptation,
            "emotion_analysis": settings.enable_emotion_analysis,
            "multilingual": settings.enable_multilingual,
            "analytics": settings.enable_analytics
        }
        
        print("✅ Feature flags status:")
        for feature, status in feature_flags.items():
            status_icon = "🟢" if status else "🔴"
            print(f"   {status_icon} {feature}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature flags error: {e}")
        return False


def test_directories_creation():
    """Test de la création des répertoires."""
    print("\n🧪 Testing directories creation...")
    
    try:
        settings = get_settings()
        
        required_dirs = [
            settings.data_dir,
            settings.models_dir,
            settings.logs_dir
        ]
        
        for directory in required_dirs:
            if directory.exists():
                print(f"✅ Directory exists: {directory}")
            else:
                print(f"❌ Directory missing: {directory}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directories creation error: {e}")
        return False


def run_all_tests():
    """Exécute tous les tests."""
    print("🚀 Neural Chat Engine - Configuration System Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Settings", test_basic_settings),
        ("Configuration Validation", test_configuration_validation),
        ("YAML Config Loader", test_yaml_config_loader),
        ("Environment Profiles", test_environment_profiles),
        ("Logging Setup", test_logging_setup),
        ("Feature Flags", test_feature_flags),
        ("Directories Creation", test_directories_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Configuration system is ready!")
        return True
    else:
        print("⚠️ Some tests failed. Check the configuration.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)