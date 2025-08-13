import os
import requests
from dotenv import load_dotenv
import logging
from time import perf_counter
import statistics
from Crypto.Cipher import AES
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if not self.token:
            logger.error("Token Telegram non configuré!")
        if not self.chat_id:
            logger.error("Chat ID Telegram non configuré!")
    def send(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Configuration Telegram incomplète")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = requests.post(url, json=payload, timeout=10)
            logger.info(f"Réponse Telegram: {response.status_code} - {response.text}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Erreur d'envoi Telegram: {e}")
            return False
    def validate_credentials(self) -> bool:
        """Vérifie que le token et chat ID sont valides"""
        if not self.token or not self.chat_id:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/getChat"
            response = requests.post(url, json={"chat_id": self.chat_id})
            return response.json().get('ok', False)
        except Exception:
            return False
class TelegramNotifierOptimized(TelegramNotifier):
    """Version optimisée avec session HTTP persistante"""
    def __init__(self):
        super().__init__()
        self.session = requests.Session()  # Session HTTP réutilisable
        self.session.headers.update({'Connection': 'keep-alive'})
    def send(self, message: str) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Configuration Telegram incomplète")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            response = self.session.post(url, json=payload, timeout=10)
            logger.info(f"Réponse Telegram (optimisée): {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Erreur d'envoi (optimisé): {e}")
            return False
    def test_latency(self, n_tests: int = 5) -> float:
        """Mesure la latence moyenne d'envoi"""
        latencies = []
        for _ in range(n_tests):
            start = perf_counter()
            self.send("Latency test ping")
            latencies.append((perf_counter() - start) * 1000)
        return statistics.mean(latencies)
    def test_success_rate(self, n_tests: int = 10) -> float:
        """Calcule le taux de réussite"""
        successes = 0
        for i in range(n_tests):
            if self.send(f"Success test {i+1}"):
                successes += 1
        return (successes / n_tests) * 100
    def send_encrypted(self, msg: str, key: str) -> bool:
        """Envoi chiffré AES-256"""
        try:
            cipher = AES.new(key.encode(), AES.MODE_EAX)
            ciphertext, tag = cipher.encrypt_and_digest(msg.encode())
            return self.send(f"ENC:{ciphertext.hex()}:{tag.hex()}")
        except Exception as e:
            self.log_error(f"Encryption failed: {str(e)}")
            return False
    def log_error(self, error_msg: str):
        """Journalisation des erreurs"""
        os.makedirs("logs", exist_ok=True)
        with open("logs/telegram_errors.log", "a") as f:
# Alias pour compatibilité ascendante
TelegramNotifier = TelegramNotifierOptimized
