"""
Reusable Chat Interface Components for Neural Chat Engine

This module provides modular, reusable components for building chat interfaces,
including message bubbles, typing indicators, file previews, confidence meters,
and other UI elements that can be used across different frontend implementations.
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import base64
import json
from pathlib import Path

from ...api.api_types import MessageRole, MessageType, ChatMessage


class ChatComponents:
    """Collection of reusable chat interface components."""

    @staticmethod
    def inject_custom_css():
        """Inject custom CSS for enhanced component styling."""
        st.markdown("""
        <style>
            .message-bubble {
                padding: 1rem 1.2rem;
                border-radius: 1rem;
                margin: 0.5rem 0;
                max-width: 80%;
                word-wrap: break-word;
                position: relative;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .message-bubble.user {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                margin-left: auto;
                border-bottom-right-radius: 0.2rem;
            }
            .message-bubble.assistant {
                background: #f8f9fa;
                color: #333;
                border: 1px solid #e9ecef;
                margin-right: auto;
                border-bottom-left-radius: 0.2rem;
            }
            .message-bubble.system {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
                margin: 0.5rem auto;
                max-width: 90%;
                text-align: center;
            }
            .message-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 0.5rem;
                vertical-align: top;
            }
            .typing-dots {
                display: inline-flex;
                gap: 0.2rem;
                align-items: center;
            }
            .typing-dot {
                width: 8px;
                height: 8px;
                background-color: #6c757d;
                border-radius: 50%;
                animation: typing-bounce 1.4s infinite ease-in-out both;
            }
            .typing-dot:nth-child(1) { animation-delay: -0.32s; }
            .typing-dot:nth-child(2) { animation-delay: -0.16s; }
            @keyframes typing-bounce {
                0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
                40% { transform: scale(1); opacity: 1; }
            }
            .confidence-bar {
                width: 100%;
                height: 6px;
                background-color: #e9ecef;
                border-radius: 3px;
                margin: 0.5rem 0;
                overflow: hidden;
            }
            .confidence-fill {
                height: 100%;
                border-radius: 3px;
                transition: width 0.3s ease, background-color 0.3s ease;
            }
            .confidence-fill.high { background-color: #28a745; }
            .confidence-fill.medium { background-color: #ffc107; }
            .confidence-fill.low { background-color: #dc3545; }
            .file-preview {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 0.5rem 0;
            }
            .file-icon {
                font-size: 2rem;
                margin-right: 0.5rem;
            }
            .personality-selector {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin: 1rem 0;
            }
            .personality-button {
                padding: 0.5rem 1rem;
                border: 2px solid #dee2e6;
                border-radius: 2rem;
                background: white;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .personality-button.active {
                border-color: #007bff;
                background: #007bff;
                color: white;
            }
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def message_bubble(
        content: str,
        role: MessageRole,
        timestamp: Optional[datetime] = None,
        message_type: MessageType = MessageType.TEXT,
        confidence: Optional[float] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Render a message bubble with proper styling.

        Args:
            content: Message content
            role: Message role (user/assistant/system)
            timestamp: Message timestamp
            message_type: Type of message content
            confidence: Model confidence score (0-1)
            attachments: List of attached files
            metadata: Additional message metadata
        """
        # Determine bubble class
        bubble_class = f"message-bubble {role.value.lower()}"

        # Create message container
        st.markdown(f'<div class="{bubble_class}">', unsafe_allow_html=True)

        # Render content based on type
        if message_type == MessageType.CODE:
            st.code(content, language='python')
        elif message_type == MessageType.MARKDOWN:
            st.markdown(content)
        else:
            st.markdown(content)

        # Attachments
        if attachments:
            ChatComponents.file_preview(attachments)

        # Confidence meter for assistant messages
        if role == MessageRole.ASSISTANT and confidence is not None:
            ChatComponents.confidence_meter(confidence)

        # Timestamp
        if timestamp:
            time_str = timestamp.strftime("%H:%M") if isinstance(timestamp, datetime) else str(timestamp)
            st.caption(f"📅 {time_str}")

        # Debug metadata
        if metadata and st.session_state.get('debug_mode', False):
            with st.expander("📊 Métadonnées"):
                st.json(metadata)

        st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def typing_indicator():
        """Render an animated typing indicator."""
        st.markdown("""
        <div style="display: flex; align-items: center; padding: 1rem;">
            <div class="message-avatar" style="background: #007bff; color: white; display: flex; align-items: center; justify-content: center;">🤖</div>
            <div class="message-bubble assistant">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span style="margin-left: 0.5rem; color: #6c757d;">L'assistant tape...</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def confidence_meter(confidence: float, show_percentage: bool = True):
        """
        Render a confidence meter.

        Args:
            confidence: Confidence score between 0 and 1
            show_percentage: Whether to show percentage text
        """
        # Determine confidence level and color
        if confidence >= 0.8:
            level = "high"
            color = "#28a745"
            label = "Élevée"
        elif confidence >= 0.6:
            level = "medium"
            color = "#ffc107"
            label = "Moyenne"
        else:
            level = "low"
            color = "#dc3545"
            label = "Faible"

        # Render meter
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill {level}" style="width: {confidence*100}%"></div>
        </div>
        """, unsafe_allow_html=True)

        if show_percentage:
            st.caption(f"🔍 Confiance: {label} ({confidence:.1%})")

    @staticmethod
    def file_preview(attachments: List[Dict[str, Any]]):
        """
        Render file preview component.

        Args:
            attachments: List of file attachment dictionaries
        """
        for attachment in attachments:
            filename = attachment.get('filename', 'unknown')
            file_type = attachment.get('type', 'unknown')
            size = attachment.get('size', 0)

            # File icon based on type
            icon_map = {
                'pdf': '📄',
                'txt': '📝',
                'py': '🐍',
                'md': '📖',
                'json': '📊',
                'image': '🖼️',
                'audio': '🎵',
                'video': '🎬'
            }

            icon = icon_map.get(file_type.split('/')[0], '📎') if '/' in file_type else icon_map.get(file_type, '📎')

            with st.container():
                st.markdown(f"""
                <div class="file-preview">
                    <span class="file-icon">{icon}</span>
                    <strong>{filename}</strong>
                    <br>
                    <small>Type: {file_type} | Taille: {ChatComponents._format_file_size(size)}</small>
                </div>
                """, unsafe_allow_html=True)

    @staticmethod
    def personality_selector(personalities: List[Dict[str, str]], current_personality: Optional[str] = None):
        """
        Render personality selector component.

        Args:
            personalities: List of personality dictionaries with 'id' and 'name' keys
            current_personality: Currently selected personality ID
        """
        st.markdown('<div class="personality-selector">', unsafe_allow_html=True)

        cols = st.columns(len(personalities))
        for i, personality in enumerate(personalities):
            with cols[i]:
                is_active = current_personality == personality['id']
                button_class = "personality-button active" if is_active else "personality-button"

                if st.button(
                    personality['name'],
                    key=f"personality_{personality['id']}",
                    help=personality.get('description', ''),
                    use_container_width=True
                ):
                    st.session_state.current_personality = personality['id']
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def theme_toggle():
        """Render theme toggle component."""
        current_theme = st.session_state.get('theme', 'light')
        new_theme = 'dark' if current_theme == 'light' else 'light'

        icon = '🌙' if current_theme == 'light' else '☀️'
        label = f'{icon} Thème {new_theme.title()}'

        if st.button(label, key='theme_toggle'):
            st.session_state.theme = new_theme
            # Apply theme (would need additional CSS for full dark mode)
            st.rerun()

    @staticmethod
    def conversation_summary(conversation_data: Dict[str, Any]):
        """
        Render conversation summary component.

        Args:
            conversation_data: Dictionary containing conversation statistics
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Messages", conversation_data.get('message_count', 0))
        with col2:
            st.metric("Durée", f"{conversation_data.get('duration_minutes', 0)}min")
        with col3:
            st.metric("Confiance moy.", f"{conversation_data.get('avg_confidence', 0):.1%}")
        with col4:
            rating = conversation_data.get('user_rating', 0)
            st.metric("Évaluation", f"{'⭐' * rating if rating else 'N/A'}")

    @staticmethod
    def export_conversation_button(conversation_data: Dict[str, Any], filename: Optional[str] = None):
        """
        Render export conversation button.

        Args:
            conversation_data: Conversation data to export
            filename: Optional custom filename
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        json_data = json.dumps(conversation_data, indent=2, ensure_ascii=False, default=str)

        st.download_button(
            label="📥 Exporter la conversation",
            data=json_data,
            file_name=filename,
            mime="application/json",
            use_container_width=True
        )

    @staticmethod
    def feedback_form(session_id: str, on_submit_callback: Optional[callable] = None):
        """
        Render feedback form component.

        Args:
            session_id: Current session ID
            on_submit_callback: Callback function to handle form submission
        """
        with st.form(key=f"feedback_{session_id}"):
            st.subheader("⭐ Évaluez cette conversation")

            rating = st.slider("Note (1-5 étoiles)", 1, 5, 3)

            feedback_text = st.text_area(
                "Commentaires (optionnel)",
                placeholder="Qu'avez-vous pensé de cette conversation ?"
            )

            categories = st.multiselect(
                "Aspects à améliorer",
                ["Rapidité", "Pertinence", "Clarté", "Créativité", "Fiabilité"],
                default=[]
            )

            submitted = st.form_submit_button("Envoyer l'évaluation")

            if submitted:
                feedback_data = {
                    'session_id': session_id,
                    'rating': rating,
                    'feedback_text': feedback_text,
                    'categories': categories,
                    'timestamp': datetime.now().isoformat()
                }

                if on_submit_callback:
                    on_submit_callback(feedback_data)
                else:
                    st.success("Merci pour votre retour ! 🎉")

    @staticmethod
    def debug_info_panel(data: Dict[str, Any], title: str = "Debug Information"):
        """
        Render debug information panel.

        Args:
            data: Data to display in debug panel
            title: Panel title
        """
        if st.session_state.get('debug_mode', False):
            with st.expander(f"🐛 {title}"):
                st.json(data)

    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return ".1f"

    @staticmethod
    def loading_spinner(text: str = "Chargement..."):
        """Render a loading spinner component."""
        with st.spinner(text):
            time.sleep(0.1)  # Small delay for visual effect

    @staticmethod
    def error_message(message: str, details: Optional[str] = None):
        """Render an error message component."""
        st.error(f"❌ {message}")
        if details:
            with st.expander("Détails de l'erreur"):
                st.code(details)

    @staticmethod
    def success_message(message: str):
        """Render a success message component."""
        st.success(f"✅ {message}")

    @staticmethod
    def info_message(message: str):
        """Render an info message component."""
        st.info(f"ℹ️ {message}")

    @staticmethod
    def warning_message(message: str):
        """Render a warning message component."""
        st.warning(f"⚠️ {message}")


class ChatLayout:
    """Layout utilities for chat interfaces."""

    @staticmethod
    def create_sidebar_navigation(pages: List[Dict[str, str]]):
        """
        Create sidebar navigation.

        Args:
            pages: List of page dictionaries with 'name' and 'icon' keys
        """
        st.sidebar.title("🧭 Navigation")

        for page in pages:
            if st.sidebar.button(f"{page['icon']} {page['name']}", use_container_width=True):
                st.session_state.current_page = page['name'].lower().replace(' ', '_')
                st.rerun()

    @staticmethod
    def create_message_container(height: str = "400px"):
        """
        Create a scrollable message container.

        Args:
            height: Container height (CSS value)
        """
        st.markdown(f"""
        <div style="
            height: {height};
            overflow-y: auto;
            padding: 1rem;
            border: 1px solid #e0e0e0;
            border-radius: 0.5rem;
            background: #fafafa;
        " id="message-container">
        """, unsafe_allow_html=True)

        return st.container()

    @staticmethod
    def create_input_area(placeholder: str = "Tapez votre message..."):
        """
        Create message input area.

        Args:
            placeholder: Input placeholder text
        """
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_area(
                "",
                placeholder=placeholder,
                height=80,
                key="message_input",
                label_visibility="collapsed"
            )

        with col2:
            send_button = st.button("📤", type="primary", use_container_width=True)

        return user_input, send_button


# Convenience functions for easy importing
def message_bubble(*args, **kwargs):
    """Convenience function for message bubble component."""
    return ChatComponents.message_bubble(*args, **kwargs)

def typing_indicator():
    """Convenience function for typing indicator."""
    return ChatComponents.typing_indicator()

def confidence_meter(*args, **kwargs):
    """Convenience function for confidence meter."""
    return ChatComponents.confidence_meter(*args, **kwargs)

def file_preview(*args, **kwargs):
    """Convenience function for file preview."""
    return ChatComponents.file_preview(*args, **kwargs)