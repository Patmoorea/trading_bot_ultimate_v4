import os
import json
import base64
import getpass
from cryptography.fernet import Fernet
from hashlib import sha256

class KeyManager:
    """
    Gère la génération, l'encryptage, le stockage, et la récupération sécurisée de clés privées.
    """

    def __init__(self, key_path="src/security/keys.enc"):
        self.key_path = key_path
        self._private_key = None

    def _get_password(self, prompt="Entrez votre mot de passe de wallet: "):
        # Pour tests automatisés, on peut injecter par variable d'env
        return os.getenv("WALLET_PWD") or getpass.getpass(prompt)

    def _get_fernet(self, pwd):
        # Derive une clé Fernet à partir d'un mot de passe utilisateur
        key = sha256(pwd.encode()).digest()
        return Fernet(base64.urlsafe_b64encode(key[:32]))

    def generate_private_key(self):
        # Génère une "clé privée" factice (pour BTC/ETH, remplacer par vrai algo)
        pk = base64.urlsafe_b64encode(os.urandom(32)).decode()
        self._private_key = pk
        return pk

    def save_private_key(self, password=None):
        if not self._private_key:
            raise Exception("Aucune clé privée à sauvegarder (utilisez generate_private_key d'abord)")
        if password is None:
            password = self._get_password()
        fernet = self._get_fernet(password)
        tok = fernet.encrypt(self._private_key.encode())
        with open(self.key_path, "wb") as f:
            f.write(tok)

    def load_private_key(self, password=None):
        if password is None:
            password = self._get_password()
        fernet = self._get_fernet(password)
        with open(self.key_path, "rb") as f:
            tok = f.read()
        self._private_key = fernet.decrypt(tok).decode()
        return self._private_key

    def sign_message(self, message):
        """
        Interface simple de signature (hash, pour démo : remplacer par signature crypto réelle pour BTC/ETH)
        """
        if not self._private_key:
            raise Exception("Aucune clé chargée")
        return sha256((self._private_key + message).encode()).hexdigest()

    def has_key(self):
        return os.path.exists(self.key_path)
