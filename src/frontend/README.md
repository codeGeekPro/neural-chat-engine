# Interface Utilisateur - Neural Chat Engine

## Vue d'ensemble

L'interface utilisateur du Neural Chat Engine fournit une expérience de chat moderne et intuitive avec des fonctionnalités avancées pour interagir avec le modèle d'IA.

## Composants

### 1. Application Streamlit (`streamlit_app.py`)

Application web complète construite avec Streamlit offrant :

#### Fonctionnalités principales
- **Messagerie en temps réel** avec indicateurs de frappe
- **Historique des conversations** avec navigation
- **Panneau de préférences utilisateur** personnalisables
- **Paramètres du modèle** (température, tokens max, etc.)
- **Mode debug** pour les développeurs
- **Téléchargement de fichiers** multimédia
- **Système d'évaluation** des réponses
- **Tableau de bord d'analyse** avec métriques

#### Interface utilisateur
- Design responsive avec thème sombre/clair
- Bulles de messages stylisées
- Indicateurs de confiance du modèle
- Barre latérale avec historique
- Panneau de paramètres extensible

### 2. Composants Réutilisables (`components/chat_interface.py`)

Bibliothèque de composants modulaires pour construire des interfaces de chat :

#### Composants disponibles
- `message_bubble()` - Bulles de messages stylisées
- `typing_indicator()` - Animation d'indicateur de frappe
- `confidence_meter()` - Barre de confiance du modèle
- `file_preview()` - Aperçu des fichiers attachés
- `personality_selector()` - Sélecteur de personnalités
- `theme_toggle()` - Bascule thème sombre/clair
- `conversation_summary()` - Résumé de conversation
- `export_conversation_button()` - Export de conversation
- `feedback_form()` - Formulaire d'évaluation
- `debug_info_panel()` - Panneau d'informations debug

## Installation

### Dépendances

```bash
pip install streamlit plotly streamlit-chat streamlit-extras pydantic-settings
```

### Configuration

Assurez-vous que l'API backend est en cours d'exécution :

```bash
# Terminal 1 - API Backend
cd /workspaces/neural-chat-engine
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Interface Streamlit
streamlit run src/frontend/streamlit_app.py
```

## Utilisation

### Lancement rapide

```bash
# Depuis la racine du projet
streamlit run src/frontend/streamlit_app.py
```

L'interface sera accessible sur `http://localhost:8501`

### Configuration des variables d'environnement

Créez un fichier `.env` dans la racine du projet :

```env
# API Configuration
API_HOST=localhost
API_PORT=8000

# Model Configuration
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database
POSTGRES_USER=neural_user
POSTGRES_PASSWORD=neural_pass
POSTGRES_DB=neural_chat_db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Fonctionnalités détaillées

### Messagerie

- **Messages texte** avec support Markdown
- **Messages code** avec coloration syntaxique
- **Messages système** pour les notifications
- **Pièces jointes** : images, documents, audio
- **Streaming** des réponses en temps réel

### Gestion des conversations

- **Création** de nouvelles conversations
- **Navigation** dans l'historique
- **Recherche** dans les conversations
- **Export** au format JSON
- **Évaluation** des réponses (1-5 étoiles)

### Personnalisation

- **Thèmes** : sombre/clair
- **Personnalités** du chatbot
- **Paramètres du modèle** :
  - Température (créativité)
  - Nombre maximum de tokens
  - Top-p et Top-k sampling

### Analyse et monitoring

- **Métriques temps réel** :
  - Nombre total de messages
  - Temps de réponse moyen
  - Taux de satisfaction
- **Graphiques d'utilisation** des modèles
- **Suivi des performances**

### Mode développeur

- **Informations de debug** détaillées
- **Logs API** en temps réel
- **Métadonnées** des messages
- **État de la session** complète

## Architecture

```
src/frontend/
├── streamlit_app.py          # Application principale
├── components/
│   └── chat_interface.py     # Composants réutilisables
└── __init__.py              # Exports du module
```

## API Integration

L'interface communique avec l'API backend via HTTP :

- `GET /health` - Statut du service
- `POST /chat/session` - Création de session
- `GET /chat/sessions` - Liste des sessions
- `POST /chat/message` - Envoi de message
- `POST /feedback` - Soumission d'évaluation

## Personnalisation

### Ajout de nouveaux composants

```python
from src.frontend.components.chat_interface import ChatComponents

# Créer un composant personnalisé
@staticmethod
def custom_component():
    # Votre logique ici
    pass

# L'ajouter à la classe ChatComponents
ChatComponents.custom_component = custom_component
```

### Extension des thèmes

Les thèmes sont gérés via CSS injecté. Modifiez la méthode `inject_custom_css()` pour ajouter vos styles.

### Nouveaux types de messages

Ajoutez de nouveaux types dans `MessageType` enum et implémentez le rendu correspondant dans `message_bubble()`.

## Déploiement

### Développement local

```bash
streamlit run src/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Production

Pour le déploiement en production, considérez :

1. **Streamlit Cloud** pour déploiement rapide
2. **Docker** pour conteneurisation
3. **Serveur dédié** avec nginx reverse proxy
4. **Authentification** pour accès sécurisé

### Configuration production

```bash
# Variables d'environnement pour production
ENVIRONMENT=production
DEBUG=false
API_HOST=your-api-domain.com
API_PORT=443
```

## Dépannage

### Problèmes courants

1. **Erreur de connexion API**
   - Vérifiez que l'API backend est démarrée
   - Contrôlez les variables `API_HOST` et `API_PORT`

2. **Messages non affichés**
   - Vérifiez la console pour les erreurs JavaScript
   - Contrôlez les permissions CORS

3. **Thème non appliqué**
   - Clear le cache Streamlit : `streamlit cache clear`
   - Redémarrez l'application

### Logs et debug

Activez le mode debug dans l'interface pour voir :
- Logs détaillés des requêtes API
- État des variables de session
- Métadonnées des messages
- Performances des composants

## Contribution

Pour contribuer à l'interface utilisateur :

1. Respectez la structure modulaire des composants
2. Ajoutez des tests pour les nouvelles fonctionnalités
3. Documentez les nouveaux composants
4. Suivez les conventions de nommage existantes

## Support

- **Documentation API** : `README_API.md`
- **Issues GitHub** pour les bugs
- **Discussions** pour les questions générales</content>
<parameter name="filePath">/workspaces/neural-chat-engine/src/frontend/README.md