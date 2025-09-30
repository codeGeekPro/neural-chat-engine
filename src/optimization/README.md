# Module d'Optimisation des Modèles

Ce module fournit des outils avancés pour l'optimisation des modèles de machine learning, visant à améliorer les performances d'inférence et réduire l'empreinte mémoire.

## Fonctionnalités Principales

### 1. Quantification des Modèles
```python
optimizer.quantize_model(model_name, quantization_type="dynamic")
```

- **Quantification Dynamique** : Optimisée pour les modèles d'inférence
- **Quantification Statique** : Pour des performances maximales sur CPU
- Réduction de la taille du modèle jusqu'à 75%
- Support des formats INT8 et FP16

### 2. Élagage des Poids (Pruning)
```python
optimizer.prune_model_weights(model, pruning_ratio=0.2)
```

- Élagage L1 non structuré
- Configuration du ratio d'élagage
- Préservation de la précision du modèle
- Réduction significative de la taille du modèle

### 3. Distillation de Connaissances
```python
optimizer.distill_knowledge(teacher_model, student_model, training_data)
```

- Transfert de connaissances du modèle professeur vers l'étudiant
- Température ajustable pour le softmax
- Équilibrage entre distillation et perte directe
- Optimisation pour les petits modèles

### 4. Export ONNX
```python
optimizer.export_to_onnx(model, export_path)
```

- Compatibilité multi-plateformes
- Support des axes dynamiques
- Vérification automatique du modèle exporté
- Optimisations ONNX intégrées

### 5. Benchmarking de Performance
```python
metrics = optimizer.benchmark_model_performance(model, test_data)
```

Métriques mesurées :
- Temps d'inférence
- Utilisation mémoire
- Taille du modèle
- Précision sur les données de test

### 6. Optimisation Automatique
```python
optimized_model = optimizer.auto_optimize_for_deployment(
    target_platform="mobile",
    model_name="my_model"
)
```

Configurations pré-définies pour :
- Déploiement mobile
- Edge computing
- Serveurs haute performance

## Utilisation

### Installation des Dépendances

```bash
pip install torch torchvision onnx onnxruntime
```

### Exemple d'Utilisation Basique

```python
from optimization.model_optimizer import ModelOptimizer

# Initialisation
optimizer = ModelOptimizer(models_registry)

# Optimisation complète pour mobile
optimized_model = optimizer.auto_optimize_for_deployment(
    "mobile",
    "my_model"
)

# Vérification des performances
metrics = optimizer.benchmark_model_performance(
    optimized_model,
    test_data
)
print(f"Temps d'inférence : {metrics.inference_time}ms")
```

### Configuration Avancée

```python
# Configuration personnalisée pour edge device
optimized_model = optimizer.auto_optimize_for_deployment(
    "edge",
    "my_model",
    performance_threshold={
        "inference_time": 100,  # ms
        "memory_usage": 512,    # MB
        "accuracy": 0.95
    }
)
```

## Meilleures Pratiques

1. **Choix de la Quantification**
   - Dynamique : Pour les modèles avec charges variables
   - Statique : Pour les performances maximales sur CPU

2. **Élagage des Poids**
   - Commencer avec un ratio faible (0.1-0.2)
   - Augmenter progressivement en surveillant la précision
   - Recalibrer si nécessaire

3. **Distillation**
   - Utiliser un modèle professeur bien entraîné
   - Ajuster la température selon la complexité
   - Équilibrer distillation et perte directe

4. **Optimisation Automatique**
   - Définir clairement la plateforme cible
   - Spécifier des seuils de performance réalistes
   - Valider sur des données représentatives

## Dépannage

### Problèmes Courants

1. **Perte de Précision Post-Quantification**
   - Vérifier le type de quantification
   - Ajuster les paramètres de calibration
   - Considérer la quantification par couche

2. **Performances Dégradées**
   - Vérifier la compatibilité matérielle
   - Optimiser la taille des batchs
   - Ajuster les paramètres d'inférence

3. **Échec d'Export ONNX**
   - Vérifier la version d'opset
   - Simplifier le modèle si nécessaire
   - Valider les formats d'entrée/sortie

## Métriques et Monitoring

Le module fournit des métriques détaillées :

```python
metrics = optimizer.benchmark_model_performance(model, test_data)
print(f"""
Performance Metrics:
- Inference Time: {metrics.inference_time}ms
- Memory Usage: {metrics.memory_usage}MB
- Model Size: {metrics.model_size}MB
- Accuracy: {metrics.accuracy:.2%}
""")