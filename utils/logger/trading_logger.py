import logging
from datetime import datetime
import os
from typing import Dict, Any
import json
class TradingLogger:
    def __init__(self, name: str = "trading_bot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # Création du dossier logs s'il n'existe pas
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Handler pour fichier avec rotation quotidienne
        file_handler = logging.FileHandler(
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(file_handler)
        # Handler pour console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(console_handler)
    def log_trade(self, trade_data: Dict[str, Any]):
        self.logger.info(f"TRADE: {json.dumps(trade_data, indent=2)}")
    def log_error(self, error_msg: str, error: Exception = None):
        if error:
            self.logger.error(f"{error_msg}: {str(error)}")
        else:
            self.logger.error(error_msg)
    def log_balance(self, balance_data: Dict[str, Any]):
        self.logger.info(f"BALANCE: {json.dumps(balance_data, indent=2)}")
    def log_order(self, order_data: Dict[str, Any]):
        self.logger.info(f"ORDER: {json.dumps(order_data, indent=2)}")
