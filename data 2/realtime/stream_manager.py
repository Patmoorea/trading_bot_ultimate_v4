"""Gestionnaire de flux de données en temps réel"""
from typing import Optional
class StreamManager:
    def __init__(self):
        self._active = False
    def start_stream(self) -> bool:
        """Démarre le flux de données"""
        self._active = True
        return self._active
    def stop_stream(self) -> bool:
        """Arrête le flux de données"""
        self._active = False
        return self._active
    @property
    def is_active(self) -> bool:
        """Vérifie si le flux est actif"""
        return self._active
