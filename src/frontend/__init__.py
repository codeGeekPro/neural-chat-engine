"""
Interface Frontend - Neural Chat Engine

Interfaces utilisateur pour le chatbot :  
- Application Streamlit pour prototypage rapide
- Interface React pour production
- Dashboard analytics et monitoring
"""

from .streamlit_app import StreamlitChatApp

__all__ = [
    "StreamlitChatApp"
]