"""
Système d'alerting intelligent pour la surveillance et l'escalade.
"""
from typing import Dict, Any, List, Callable
import logging
import time

class AlertingSystem:
    def __init__(self):
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: List[Dict[str, Any]] = []
        self.escalation_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.logger = logging.getLogger(__name__)

    def add_alert_rule(self, rule: Dict[str, Any]):
        """Ajoute une règle d'alerte (type, seuil, action, etc.)"""
        self.alert_rules.append(rule)
        self.logger.info(f"Alert rule added: {rule}")

    def check_metrics(self, metrics: Dict[str, Any]):
        """Vérifie les métriques et déclenche les alertes si nécessaire."""
        for rule in self.alert_rules:
            metric = rule.get('metric')
            threshold = rule.get('threshold')
            alert_type = rule.get('type')
            value = metrics.get(metric)
            if value is not None and value >= threshold:
                alert = {
                    'type': alert_type,
                    'metric': metric,
                    'value': value,
                    'threshold': threshold,
                    'timestamp': time.time()
                }
                self.active_alerts.append(alert)
                self.logger.warning(f"Alert triggered: {alert}")
                self.handle_escalation(alert)

    def handle_escalation(self, alert: Dict[str, Any]):
        """Gère l'escalade intelligente des alertes."""
        for callback in self.escalation_callbacks:
            callback(alert)
        self.logger.info(f"Escalation handled for alert: {alert}")

    def add_escalation_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Ajoute une fonction d'escalade (ex: notification, SMS, Slack)."""
        self.escalation_callbacks.append(callback)
        self.logger.info("Escalation callback added.")

    def filter_alerts(self):
        """Filtre les alertes pour éviter le bruit (smart filtering)."""
        # Exemple : ne garder que les alertes critiques ou non répétées
        filtered = []
        seen = set()
        for alert in self.active_alerts:
            key = (alert['type'], alert['metric'])
            if key not in seen and alert['value'] >= alert['threshold']:
                filtered.append(alert)
                seen.add(key)
        self.active_alerts = filtered
        self.logger.info(f"Alerts filtered: {filtered}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retourne la liste des alertes actives."""
        return self.active_alerts
