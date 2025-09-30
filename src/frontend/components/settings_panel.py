"""
Component pour le panneau de configuration de l'interface utilisateur.
"""
import streamlit as st
from typing import Dict, Any, Callable

class SettingsPanel:
    def __init__(self, on_settings_change: Callable[[Dict[str, Any]], None]):
        """
        Initialise le panneau de configuration.
        
        Args:
            on_settings_change: Callback appelé lorsque les paramètres sont modifiés
        """
        self.on_settings_change = on_settings_change
        self.settings = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "streaming": True
        }

    def render(self):
        """Affiche le panneau de configuration dans Streamlit."""
        with st.sidebar:
            st.header("Paramètres")
            
            model = st.selectbox(
                "Modèle",
                ["gpt-3.5-turbo", "gpt-4"],
                index=0
            )
            
            temperature = st.slider(
                "Température",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Contrôle la créativité des réponses"
            )
            
            max_tokens = st.number_input(
                "Tokens maximum",
                min_value=100,
                max_value=4000,
                value=1000,
                step=100,
                help="Nombre maximum de tokens dans la réponse"
            )
            
            streaming = st.toggle(
                "Streaming",
                value=True,
                help="Afficher les réponses au fur et à mesure"
            )
            
            # Met à jour les paramètres si modifiés
            new_settings = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "streaming": streaming
            }
            
            if new_settings != self.settings:
                self.settings = new_settings
                self.on_settings_change(new_settings)