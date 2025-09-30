# Guide de Tests & Déploiement Neural Chat Engine

## 1. Tests Locaux

### 1.1. Tests unitaires et de performance
```bash
pytest tests/ --maxfail=2 --disable-warnings
pytest tests/test_model_performance.py
```

### 1.2. Linting & Formatage
```bash
flake8 src/ tests/
black --check src/ tests/
```

### 1.3. Sécurité
```bash
bandit -r src/ -ll
```

### 1.4. Couverture des tests
```bash
coverage run -m pytest
coverage report -m
```

## 2. CI/CD GitHub Actions

- Le pipeline CI/CD est défini dans `.github/workflows/ci-cd.yml`
- Il inclut : tests, lint, scan sécurité, validation modèle, build Docker, déploiement, rollback
- Les résultats sont visibles dans l’onglet Actions du repo GitHub

## 3. Déploiement Local (Docker)

### 3.1. Build et lancement des services
```bash
docker-compose up --build
```

### 3.2. Accès aux services
- API : http://localhost:8000
- Frontend : http://localhost:8501
- Grafana : http://localhost:3000 (admin/admin_pass_2023)
- Prometheus : http://localhost:9090
- Nginx : http://localhost

### 3.3. Monitoring & Observabilité
- Les métriques sont exposées sur `/metrics` (API)
- Prometheus scrape automatiquement l’API et les workers
- Dashboards Grafana préconfigurés dans `docker/grafana/dashboards`

## 4. Déploiement Production

- Adapter les variables d’environnement dans `docker-compose.yml` et `.env`
- Utiliser des secrets pour les mots de passe et clés API
- Configurer Nginx pour HTTPS (voir `docker/nginx/certs`)
- Surveiller les logs et les métriques via Grafana/Prometheus

## 5. Rollback & Troubleshooting

- En cas d’échec du déploiement, le pipeline CI/CD effectue un rollback automatique
- Vérifier les logs des services avec :
```bash
docker-compose logs api
```
- Vérifier la santé des services avec :
```bash
docker-compose ps
```
- Pour les problèmes de dépendances, vérifier le dossier `requirements/`

## 6. Bonnes pratiques
- Toujours lancer les tests avant un déploiement
- Surveiller les métriques de performance et d’erreur
- Mettre à jour la documentation après chaque modification majeure
- Utiliser des branches pour les évolutions et les correctifs

---

Pour toute question ou problème, consultez la documentation dans `docs/` ou ouvrez une issue sur GitHub.
