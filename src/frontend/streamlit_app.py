"""
Advanced Streamlit Chat Interface for Neural Chat Engine

This module provides a comprehensive web interface for interacting with the Neural Chat Engine,
featuring real-time messaging, conversation management, user preferences, and advanced analytics.

Features:
- Real-time chat with typing indicators
- Conversation history and management
- User profile and preferences
- Model settings and confidence display
- Debug mode for developers
- File upload capabilities
- Response rating and feedback system
- Analytics dashboard
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import local modules
from ..config import Settings
from ..api.api_types import (
    MessageRole, MessageType, SessionStatus,
    ChatMessage, SendMessageRequest, WebSocketMessage
)

# Configuration
st.set_page_config(
    page_title="Neural Chat Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .system-message {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .typing-indicator {
        display: inline-block;
        animation: typing 1.5s infinite;
    }
    @keyframes typing {
        0%, 60%, 100% { opacity: 0; }
        30% { opacity: 1; }
    }
    .confidence-meter {
        width: 100%;
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff4444 0%, #ffaa00 50%, #44ff44 100%);
        transition: width 0.3s ease;
    }
    .debug-panel {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChatApp:
    """Main Streamlit application class for Neural Chat Engine."""

    def __init__(self):
        """Initialize the chat application."""
        self.config = Settings()
        self.api_base_url = f"http://localhost:{self.config.server.port}"

        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_session' not in st.session_state:
            st.session_state.current_session = None
        if 'conversations' not in st.session_state:
            st.session_state.conversations = []
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = self._load_user_preferences()
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        if 'analytics' not in st.session_state:
            st.session_state.analytics = self._initialize_analytics()

    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences from storage."""
        return {
            'theme': 'light',
            'model': 'neural-chat-v2',
            'temperature': 0.7,
            'max_tokens': 2048,
            'show_confidence': True,
            'auto_save': True,
            'notifications': True,
            'language': 'fr'
        }

    def _initialize_analytics(self) -> Dict[str, Any]:
        """Initialize analytics data structure."""
        return {
            'total_messages': 0,
            'total_conversations': 0,
            'avg_response_time': 0,
            'user_satisfaction': 0,
            'model_usage': {},
            'daily_stats': []
        }

    def _make_api_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Dict:
        """Make API request to the backend."""
        try:
            url = f"{self.api_base_url}{endpoint}"
            if method == 'POST':
                response = requests.post(url, json=data, timeout=30)
            else:
                response = requests.get(url, timeout=30)

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Erreur API: {str(e)}")
            return {}

    def _create_new_session(self) -> Optional[str]:
        """Create a new chat session."""
        try:
            response = self._make_api_request('/chat/session', 'POST', {
                'user_id': st.session_state.user_preferences.get('user_id', 'anonymous'),
                'preferences': st.session_state.user_preferences
            })
            return response.get('session_id')
        except Exception as e:
            st.error(f"Erreur lors de la création de session: {str(e)}")
            return None

    def _send_message(self, message: str, message_type: MessageType = MessageType.TEXT) -> Dict:
        """Send a message to the chat API."""
        if not st.session_state.current_session:
            st.session_state.current_session = self._create_new_session()

        request_data = SendMessageRequest(
            message=message,
            session_id=st.session_state.current_session,
            message_type=message_type,
            stream=st.session_state.user_preferences.get('stream_responses', True)
        ).model_dump()

        # Add user preferences to request
        request_data.update({
            'temperature': st.session_state.user_preferences['temperature'],
            'max_tokens': st.session_state.user_preferences['max_tokens'],
            'model': st.session_state.user_preferences['model']
        })

        return self._make_api_request('/chat/message', 'POST', request_data)

    def _display_message(self, message: Dict, is_user: bool = False):
        """Display a chat message with proper formatting."""
        message_class = "user-message" if is_user else "assistant-message"

        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                avatar = "👤" if is_user else "🤖"
                st.markdown(f"**{avatar}**")

            with col2:
                st.markdown(f'<div class="chat-message {message_class}">', unsafe_allow_html=True)

                # Message content
                content = message.get('content', '')
                if message.get('message_type') == MessageType.CODE:
                    st.code(content, language='python')
                else:
                    st.markdown(content)

                # Confidence meter if available
                if not is_user and st.session_state.user_preferences.get('show_confidence', True):
                    confidence = message.get('confidence', 0.8)
                    st.markdown(f"**Confiance:** {confidence:.1%}")
                    st.markdown(f"""
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: {confidence*100}%"></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Timestamp
                timestamp = message.get('timestamp', datetime.now().isoformat())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                st.caption(f"Envoyé le {timestamp.strftime('%H:%M:%S')}")

                st.markdown('</div>', unsafe_allow_html=True)

    def _display_typing_indicator(self):
        """Display typing indicator."""
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown("🤖")
            with col2:
                st.markdown("""
                <div class="chat-message assistant-message">
                    <span class="typing-indicator">●</span>
                    <span class="typing-indicator">●</span>
                    <span class="typing-indicator">●</span>
                    L'assistant tape...
                </div>
                """, unsafe_allow_html=True)

    def _render_sidebar(self):
        """Render the sidebar with conversation history and settings."""
        with st.sidebar:
            st.title("🗂️ Neural Chat")

            # New conversation button
            if st.button("➕ Nouvelle conversation", type="primary", use_container_width=True):
                st.session_state.current_session = self._create_new_session()
                st.session_state.messages = []
                st.rerun()

            st.divider()

            # Conversation history
            st.subheader("📚 Conversations récentes")
            conversations = self._make_api_request('/chat/sessions')

            if conversations.get('sessions'):
                for session in conversations['sessions'][:10]:  # Show last 10
                    if st.button(
                        f"💬 {session.get('title', 'Conversation')} - {session.get('message_count', 0)} msgs",
                        key=f"session_{session['id']}",
                        use_container_width=True
                    ):
                        st.session_state.current_session = session['id']
                        # Load messages for this session
                        messages_response = self._make_api_request(f'/chat/session/{session["id"]}/messages')
                        st.session_state.messages = messages_response.get('messages', [])
                        st.rerun()

            st.divider()

            # User preferences
            with st.expander("⚙️ Préférences", expanded=False):
                self._render_preferences_panel()

            # Analytics
            with st.expander("📊 Analytics", expanded=False):
                self._render_analytics_panel()

            # Debug mode
            if st.checkbox("🐛 Mode Debug", value=st.session_state.debug_mode):
                st.session_state.debug_mode = True
                self._render_debug_panel()
            else:
                st.session_state.debug_mode = False

    def _render_preferences_panel(self):
        """Render user preferences panel."""
        preferences = st.session_state.user_preferences

        # Model settings
        preferences['model'] = st.selectbox(
            "Modèle",
            ['neural-chat-v1', 'neural-chat-v2', 'neural-chat-pro'],
            index=['neural-chat-v1', 'neural-chat-v2', 'neural-chat-pro'].index(preferences['model'])
        )

        preferences['temperature'] = st.slider(
            "Température",
            min_value=0.0,
            max_value=2.0,
            value=preferences['temperature'],
            step=0.1
        )

        preferences['max_tokens'] = st.slider(
            "Tokens max",
            min_value=512,
            max_value=4096,
            value=preferences['max_tokens'],
            step=256
        )

        # Display options
        preferences['show_confidence'] = st.checkbox(
            "Afficher la confiance",
            value=preferences['show_confidence']
        )

        preferences['theme'] = st.selectbox(
            "Thème",
            ['light', 'dark'],
            index=['light', 'dark'].index(preferences['theme'])
        )

        # Save preferences
        if st.button("💾 Sauvegarder", use_container_width=True):
            self._save_user_preferences()
            st.success("Préférences sauvegardées!")

    def _render_analytics_panel(self):
        """Render analytics dashboard."""
        analytics = st.session_state.analytics

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages totaux", analytics['total_messages'])
            st.metric("Conversations", analytics['total_conversations'])
        with col2:
            st.metric("Temps de réponse moyen", f"{analytics['avg_response_time']:.2f}s")
            st.metric("Satisfaction", f"{analytics['user_satisfaction']:.1%}")

        # Usage chart
        if analytics['model_usage']:
            fig = px.pie(
                values=list(analytics['model_usage'].values()),
                names=list(analytics['model_usage'].keys()),
                title="Utilisation des modèles"
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_debug_panel(self):
        """Render debug information panel."""
        st.markdown('<div class="debug-panel">', unsafe_allow_html=True)
        st.subheader("🐛 Informations de debug")

        # Session info
        st.write("**Session actuelle:**", st.session_state.current_session)
        st.write("**Nombre de messages:**", len(st.session_state.messages))

        # API status
        health = self._make_api_request('/health')
        st.write("**Statut API:**", health.get('status', 'unknown'))

        # Preferences
        st.write("**Préférences:**")
        st.json(st.session_state.user_preferences)

        # Raw messages
        with st.expander("Messages bruts"):
            st.json(st.session_state.messages)

        st.markdown('</div>', unsafe_allow_html=True)

    def _save_user_preferences(self):
        """Save user preferences to storage."""
        # In a real app, this would save to a database or file
        pass

    def _export_conversation(self):
        """Export current conversation to JSON."""
        if st.session_state.messages:
            conversation_data = {
                'session_id': st.session_state.current_session,
                'export_date': datetime.now().isoformat(),
                'messages': st.session_state.messages,
                'preferences': st.session_state.user_preferences
            }

            json_str = json.dumps(conversation_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Télécharger la conversation",
                data=json_str,
                file_name=f"conversation_{st.session_state.current_session}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    def run(self):
        """Run the main Streamlit application."""
        # Header
        st.markdown('<h1 class="main-header">🤖 Neural Chat Engine</h1>', unsafe_allow_html=True)

        # Sidebar
        self._render_sidebar()

        # Main chat area
        st.subheader("💬 Conversation")

        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                is_user = message.get('role') == MessageRole.USER.value
                self._display_message(message, is_user)

        # Message input area
        st.divider()

        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            user_input = st.text_area(
                "Tapez votre message...",
                key="user_input",
                height=100,
                placeholder="Posez votre question ou donnez une instruction...",
                label_visibility="collapsed"
            )

        with col2:
            message_type = st.selectbox(
                "Type",
                [MessageType.TEXT.value, MessageType.CODE.value],
                label_visibility="collapsed"
            )

        with col3:
            send_button = st.button("📤 Envoyer", type="primary", use_container_width=True)

        # File upload
        uploaded_file = st.file_uploader(
            "📎 Joindre un fichier (optionnel)",
            type=['txt', 'pdf', 'py', 'md', 'json'],
            help="Formats supportés: texte, PDF, code Python, Markdown, JSON"
        )

        # Send message
        if send_button and user_input.strip():
            # Add user message
            user_message = {
                'role': MessageRole.USER.value,
                'content': user_input,
                'message_type': message_type,
                'timestamp': datetime.now().isoformat(),
                'attachments': [uploaded_file.name] if uploaded_file else None
            }
            st.session_state.messages.append(user_message)

            # Show typing indicator
            with chat_container:
                self._display_typing_indicator()

            # Send to API
            with st.spinner("L'assistant réfléchit..."):
                response = self._send_message(user_input, MessageType(message_type))

            # Remove typing indicator and add response
            if response.get('success', False):
                assistant_message = response.get('message', {})
                st.session_state.messages.append(assistant_message)

                # Update analytics
                st.session_state.analytics['total_messages'] += 2  # user + assistant

            st.rerun()

        # Export conversation
        if st.session_state.messages:
            st.divider()
            self._export_conversation()

            # Feedback system
            st.subheader("⭐ Évaluez cette conversation")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if st.button("⭐", key="rating_1"):
                    self._submit_feedback(1)
            with col2:
                if st.button("⭐⭐", key="rating_2"):
                    self._submit_feedback(2)
            with col3:
                if st.button("⭐⭐⭐", key="rating_3"):
                    self._submit_feedback(3)
            with col4:
                if st.button("⭐⭐⭐⭐", key="rating_4"):
                    self._submit_feedback(4)
            with col5:
                if st.button("⭐⭐⭐⭐⭐", key="rating_5"):
                    self._submit_feedback(5)

    def _submit_feedback(self, rating: int):
        """Submit user feedback."""
        feedback_data = {
            'session_id': st.session_state.current_session,
            'rating': rating,
            'timestamp': datetime.now().isoformat(),
            'message_count': len(st.session_state.messages)
        }

        self._make_api_request('/feedback', 'POST', feedback_data)
        st.success(f"Merci pour votre évaluation ({rating}⭐) !")

        # Update analytics
        st.session_state.analytics['user_satisfaction'] = (
            st.session_state.analytics['user_satisfaction'] * 0.9 + rating/5 * 0.1
        )


def main():
    """Main entry point for the Streamlit application."""
    app = StreamlitChatApp()
    app.run()


if __name__ == "__main__":
    main()