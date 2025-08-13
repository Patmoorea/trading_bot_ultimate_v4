import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging
class NotificationManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.notifications = []
        self._initialize_store()
    def _initialize_store(self):
        """Initialise le stockage des notifications"""
        if 'notifications' not in st.session_state:
            st.session_state.notifications = []
    def add_alert(self, 
                  message: str,
                  level: str = "info",
                  expiry: Optional[datetime] = None):
        """Ajoute une nouvelle alerte"""
        alert = {
            'message': message,
            'level': level,
            'timestamp': datetime.utcnow(),
            'expiry': expiry,
            'read': False
        }
        st.session_state.notifications.append(alert)
    def render_notifications(self):
        """Affiche les notifications dans le dashboard"""
        if st.session_state.notifications:
            with st.sidebar:
                st.subheader("🔔 Notifications")
                # Filtrer les notifications expirées
                active_notifications = [
                    n for n in st.session_state.notifications 
                    if not n['expiry'] or n['expiry'] > datetime.utcnow()
                ]
                for idx, notif in enumerate(active_notifications):
                    # Choisir le style selon le niveau
                    if notif['level'] == 'error':
                        st.error(notif['message'])
                    elif notif['level'] == 'warning':
                        st.warning(notif['message'])
                    elif notif['level'] == 'success':
                        st.success(notif['message'])
                    else:
                        st.info(notif['message'])
                    # Timestamp
                    st.caption(
                    )
                    # Bouton pour marquer comme lu
                    if not notif['read']:
                        if st.button("Marquer comme lu", key=f"read_{idx}"):
                            notif['read'] = True
                    st.markdown("---")
