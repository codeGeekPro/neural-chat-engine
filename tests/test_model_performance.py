import pytest
import time
import psutil
import numpy as np
from src.core.model_manager import ModelManager
from src.utils.metrics import calculate_metrics

class TestModelPerformance:
    def setup_method(self):
        """Initialize test environment before each test"""
        self.model_manager = ModelManager()
        self.test_samples = self._load_test_samples()

    def _load_test_samples(self):
        """Charger les échantillons de test depuis le répertoire de données"""
        return [
            ("Bonjour, comment puis-je vous aider ?", "salutation"),
            ("Je voudrais réserver un restaurant", "reservation"),
            ("Quel temps fera-t-il demain ?", "meteo"),
            ("Merci beaucoup et au revoir", "conclusion")
        ]

    def test_latency_benchmarks(self):
        """Test de latence pour différents modèles"""
        models = self.model_manager.get_available_models()
        
        for model_name, model in models.items():
            latencies = []
            
            # Mesurer la latence sur plusieurs inférences
            for text, _ in self.test_samples:
                start_time = time.time()
                _ = model.predict(text)
                latency = time.time() - start_time
                latencies.append(latency)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            
            # Vérifier les seuils de performance
            assert avg_latency < 0.5, f"Latence moyenne trop élevée pour {model_name}"
            assert p95_latency < 1.0, f"P95 latence trop élevée pour {model_name}"

    def test_accuracy_measurements(self):
        """Test de précision des modèles"""
        models = self.model_manager.get_available_models()
        
        for model_name, model in models.items():
            predictions = []
            true_labels = []
            
            for text, label in self.test_samples:
                pred = model.predict(text)
                predictions.append(pred)
                true_labels.append(label)
            
            metrics = calculate_metrics(true_labels, predictions)
            
            # Vérifier les métriques de performance
            assert metrics['accuracy'] >= 0.85
            assert metrics['f1_score'] >= 0.80
            assert metrics['precision'] >= 0.80
            assert metrics['recall'] >= 0.80

    def test_memory_usage(self):
        """Test de l'utilisation mémoire des modèles"""
        models = self.model_manager.get_available_models()
        process = psutil.Process()
        
        for model_name, model in models.items():
            # Mesurer l'utilisation mémoire avant
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Charger et utiliser le modèle
            _ = model.predict(self.test_samples[0][0])
            
            # Mesurer l'utilisation mémoire après
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_usage = mem_after - mem_before
            
            # Vérifier la consommation mémoire
            assert mem_usage < 1024, f"Utilisation mémoire excessive pour {model_name}"

    def test_model_comparison(self):
        """Tests comparatifs entre différents modèles"""
        models = self.model_manager.get_available_models()
        results = {}
        
        for model_name, model in models.items():
            # Mesurer performances
            start_time = time.time()
            predictions = []
            
            for text, _ in self.test_samples:
                pred = model.predict(text)
                predictions.append(pred)
            
            execution_time = time.time() - start_time
            
            # Calculer métriques
            metrics = calculate_metrics(
                [label for _, label in self.test_samples],
                predictions
            )
            
            results[model_name] = {
                'accuracy': metrics['accuracy'],
                'latency': execution_time / len(self.test_samples),
                'memory': psutil.Process().memory_info().rss / 1024 / 1024
            }
        
        # Comparer les modèles
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        assert best_model[1]['accuracy'] >= 0.85, "Aucun modèle n'atteint la précision requise"

    def test_ab_testing(self):
        """Framework de tests A/B pour les modèles"""
        model_a = self.model_manager.get_model("model_a")
        model_b = self.model_manager.get_model("model_b")
        
        metrics_a = self._evaluate_model(model_a)
        metrics_b = self._evaluate_model(model_b)
        
        # Comparer les performances
        assert abs(metrics_a['accuracy'] - metrics_b['accuracy']) < 0.1, \
            "Différence de performance significative entre les modèles"
        
        # Vérifier que les deux modèles répondent aux exigences minimales
        for metrics in [metrics_a, metrics_b]:
            assert metrics['accuracy'] >= 0.85
            assert metrics['latency'] < 0.5
            assert metrics['memory'] < 1024

    def _evaluate_model(self, model):
        """Évaluer un modèle sur plusieurs métriques"""
        start_time = time.time()
        predictions = []
        
        for text, _ in self.test_samples:
            pred = model.predict(text)
            predictions.append(pred)
        
        execution_time = time.time() - start_time
        
        metrics = calculate_metrics(
            [label for _, label in self.test_samples],
            predictions
        )
        
        return {
            'accuracy': metrics['accuracy'],
            'latency': execution_time / len(self.test_samples),
            'memory': psutil.Process().memory_info().rss / 1024 / 1024
        }