import numpy as np
import lz4.frame
from collections import deque

class CircularBuffer:
    def __init__(self, max_size=1000):
        """
        Initialise le buffer circulaire avec compression LZ4
        
        Args:
            max_size (int): Taille maximale du buffer
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.compressed_data = {}

    def update(self, data):
        """
        Met à jour le buffer avec de nouvelles données
        
        Args:
            data (dict): Données de marché à stocker
        """
        # Compression des données avec LZ4
        compressed = lz4.frame.compress(str(data).encode())
        self.compressed_data = compressed
        
        # Ajout au buffer
        self.buffer.append(data)

    def get_latest(self, n=1):
        """
        Récupère les n dernières entrées du buffer
        
        Args:
            n (int): Nombre d'entrées à récupérer
            
        Returns:
            list: Les n dernières entrées
        """
        return list(self.buffer)[-n:]

    def get_compressed(self):
        """
        Récupère les données compressées
        
        Returns:
            bytes: Données compressées en LZ4
        """
        return self.compressed_data
import numpy as np
import lz4.frame
from collections import deque
import logging

class CircularBuffer:
    def __init__(self, maxlen=1000, compression=True):
        """
        Initialise un buffer circulaire avec compression LZ4
        
        Args:
            maxlen (int): Taille maximale du buffer
            compression (bool): Active/désactive la compression
        """
        self.buffer = deque(maxlen=maxlen)
        self.compression = compression
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Buffer circulaire initialisé (taille max: {maxlen}, compression: {compression})")

    def add(self, data):
        """
        Ajoute des données au buffer avec compression optionnelle
        
        Args:
            data (dict/array): Données à ajouter
        """
        try:
            if self.compression:
                # Conversion en bytes pour compression
                data_bytes = np.array(data).tobytes()
                compressed = lz4.frame.compress(data_bytes)
                self.buffer.append(compressed)
            else:
                self.buffer.append(data)
                
            self.logger.debug(f"Données ajoutées au buffer (taille actuelle: {len(self.buffer)})")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout au buffer: {e}")

    def get_latest(self, n=1):
        """
        Récupère les n derniers éléments du buffer
        
        Args:
            n (int): Nombre d'éléments à récupérer
            
        Returns:
            list: Liste des derniers éléments
        """
        try:
            if n > len(self.buffer):
                n = len(self.buffer)
                
            if self.compression:
                result = []
                for i in range(-n, 0):
                    decompressed = lz4.frame.decompress(self.buffer[i])
                    data = np.frombuffer(decompressed).reshape(-1)
                    result.append(data)
                return result
            else:
                return list(self.buffer)[-n:]
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return []

    def clear(self):
        """Vide le buffer"""
        self.buffer.clear()
        self.logger.info("Buffer vidé")

    @property
    def size(self):
        """Retourne la taille actuelle du buffer"""
        return len(self.buffer)

    def get_memory_usage(self):
        """
        Calcule l'utilisation mémoire du buffer
        
        Returns:
            float: Taille en MB
        """
        try:
            total_size = sum(len(item) for item in self.buffer)
            size_mb = total_size / (1024 * 1024)
            self.logger.info(f"Utilisation mémoire: {size_mb:.2f} MB")
            return size_mb
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la mémoire: {e}")
            return 0
