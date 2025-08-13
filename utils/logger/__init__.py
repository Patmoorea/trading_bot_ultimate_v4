import os
import logging
from datetime import datetime

def get_logger(name=__name__):
    """Configure et retourne un logger"""
    # Création du logger
    logger = logging.getLogger(name)
    
    # Si le logger est déjà configuré, on le retourne directement
    if logger.handlers:
        return logger
    
    # Configuration du niveau de log
    logger.setLevel(logging.INFO)
    
    # Format du log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour le fichier
    try:
        # Création du dossier logs s'il n'existe pas
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Création du nom de fichier avec la date
        log_filename = f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
        file_path = os.path.join(logs_dir, log_filename)
        
        # Ajout du handler de fichier
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.error(f"Erreur lors de la configuration du fichier de log: {e}")
    
    return logger
