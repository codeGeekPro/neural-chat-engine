"""
Component pour l'en-tête de l'interface utilisateur.
"""
import streamlit as st

class Header:
    def __init__(self, title: str = "Neural Chat Engine"):
        """
        Initialise le composant d'en-tête.
        
        Args:
            title: Le titre principal à afficher
        """
        self.title = title

    def render(self):
        """Affiche l'en-tête dans Streamlit."""
        st.title(self.title)
        
        st.markdown("""
        Bienvenue dans Neural Chat Engine, une interface de chat alimentée par l'IA.
        
        ---
        """)
        
        # Informations sur la version et le statut
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("📡 Modèle: Actif")
        
        with col2:
            st.success("🔄 API: Connectée")
            
        with col3:
            st.warning("💾 Cache: Activé")
            
        st.markdown("---")