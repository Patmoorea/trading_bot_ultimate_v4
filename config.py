import os
from dotenv import load_dotenv
load_dotenv()
class Config:
    # Test Mode
    IS_TEST = os.getenv('IS_TEST', 'false').lower() == 'true'
    # Telegram Configuration
    _TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    _TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    # Exchange Configuration
    _EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')
    _EXCHANGE_API_SECRET = os.getenv('EXCHANGE_API_SECRET')
    # AI Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/')
    # Performance Configuration
    PERFORMANCE_LOG_PATH = os.getenv('PERFORMANCE_LOG_PATH', 'logs/performance/')
    @classmethod
    def validate(cls):
        if cls.IS_TEST:
            return
        required_vars = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
            'EXCHANGE_API_KEY',
            'EXCHANGE_API_SECRET'
        ]
        for var in required_vars:
            if not getattr(cls, var.lower()):
                raise ValueError(f"Missing required environment variable: {var}")
    @classmethod
    def telegram_bot_token(cls):
        return cls._TELEGRAM_BOT_TOKEN if not cls.IS_TEST else 'test_token'
    @classmethod
    def telegram_chat_id(cls):
        return cls._TELEGRAM_CHAT_ID if not cls.IS_TEST else 'test_chat_id'
    @classmethod
    def exchange_api_key(cls):
        return cls._EXCHANGE_API_KEY if not cls.IS_TEST else 'test_key'
    @classmethod
    def exchange_api_secret(cls):
        return cls._EXCHANGE_API_SECRET if not cls.IS_TEST else 'test_secret'
