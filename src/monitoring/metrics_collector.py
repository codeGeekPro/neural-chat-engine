"""
Système de collecte de métriques pour monitoring et observabilité avancée.
"""
from prometheus_client import Summary, Gauge, Counter, Histogram
from typing import Dict, Any, List
import time
import psutil
import logging

class MetricsCollector:
    def __init__(self, prometheus_client):
        self.prometheus_client = prometheus_client
        # Métriques Prometheus
        self.response_time = Summary('response_time_seconds', 'Temps de réponse par endpoint', ['endpoint'])
        self.model_accuracy = Gauge('model_accuracy', 'Précision du modèle', ['model_name'])
        self.user_satisfaction = Gauge('user_satisfaction', 'Note de satisfaction utilisateur', ['session_id'])
        self.error_rate = Counter('error_rate', 'Nombre d’erreurs par endpoint', ['endpoint'])
        self.model_drift = Gauge('model_drift_score', 'Score de drift du modèle', ['model_name'])
        self.cpu_usage = Gauge('cpu_usage_percent', 'Utilisation CPU (%)')
        self.memory_usage = Gauge('memory_usage_mb', 'Utilisation mémoire (MB)')
        self.business_metric = Gauge('business_metric', 'Métrique métier personnalisée', ['metric_name'])
        self.logger = logging.getLogger(__name__)

    def track_response_time(self, endpoint: str, duration: float):
        self.response_time.labels(endpoint=endpoint).observe(duration)
        self.logger.info(f"Response time tracked: {endpoint} - {duration}s")

    def track_model_accuracy(self, model_name: str, accuracy_score: float):
        self.model_accuracy.labels(model_name=model_name).set(accuracy_score)
        self.logger.info(f"Model accuracy tracked: {model_name} - {accuracy_score}")

    def track_user_satisfaction(self, session_id: str, rating: float):
        self.user_satisfaction.labels(session_id=session_id).set(rating)
        self.logger.info(f"User satisfaction tracked: {session_id} - {rating}")

    def track_conversation_metrics(self, conversation_data: Dict[str, Any]):
        # Exemple : durée, nombre de messages, taux de résolution
        duration = conversation_data.get('duration', 0)
        messages = conversation_data.get('messages', 0)
        resolved = conversation_data.get('resolved', False)
        self.business_metric.labels(metric_name='conversation_duration').set(duration)
        self.business_metric.labels(metric_name='conversation_messages').set(messages)
        self.business_metric.labels(metric_name='conversation_resolved').set(int(resolved))
        self.logger.info(f"Conversation metrics tracked: {conversation_data}")

    def detect_model_drift(self, current_performance: float, baseline: float, model_name: str):
        drift_score = abs(current_performance - baseline)
        self.model_drift.labels(model_name=model_name).set(drift_score)
        self.logger.info(f"Model drift detected: {model_name} - {drift_score}")
        return drift_score

    def track_error(self, endpoint: str):
        self.error_rate.labels(endpoint=endpoint).inc()
        self.logger.warning(f"Error tracked: {endpoint}")

    def track_resource_usage(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().used / (1024 * 1024)
        self.cpu_usage.set(cpu)
        self.memory_usage.set(mem)
        self.logger.info(f"Resource usage tracked: CPU={cpu}%, MEM={mem}MB")

    def generate_performance_report(self, time_period: str = '24h') -> Dict[str, Any]:
        # Exemple simplifié : retourne les dernières valeurs des métriques
        report = {
            'response_time': self.response_time.collect(),
            'model_accuracy': self.model_accuracy.collect(),
            'user_satisfaction': self.user_satisfaction.collect(),
            'error_rate': self.error_rate.collect(),
            'model_drift': self.model_drift.collect(),
            'cpu_usage': self.cpu_usage.collect(),
            'memory_usage': self.memory_usage.collect(),
            'business_metric': self.business_metric.collect()
        }
        self.logger.info(f"Performance report generated for {time_period}")
        return report
