import logging
import sys
from pathlib import Path
class FrenchFormatter(logging.Formatter):
    def format(self, record):
        # Traduction des niveaux de log
        level_translations = {
            'DEBUG': 'DÉBOGAGE',
            'INFO': 'INFO',
            'WARNING': 'ATTENTION',
            'ERROR': 'ERREUR',
            'CRITICAL': 'CRITIQUE'
        }
        record.levelname = level_translations.get(record.levelname, record.levelname)
        return super().format(record)
# Configuration du logger
logger = logging.getLogger('bot_arbitrage')
logger.setLevel(logging.INFO)
# Format des logs en français
formatter = FrenchFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Handler pour la console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# Handler pour les fichiers logs
log_file = Path(__file__).parent.parent / 'logs' / 'arbitrage.log'
log_file.parent.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
def get_logger():
    return logger
# Ajout en tête de fichier
import os
# Modification après getLogger
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
