import logging
from datetime import datetime
import os
class AdvancedLogger:
    def __init__(self, name="trading_bot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        # Configuration du format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Handler pour fichier
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        # Handler pour console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    def log_trade(self, trade_info):
        self.logger.info(f"TRADE: {trade_info}")
    def log_error(self, error):
        self.logger.error(f"ERROR: {error}")
    def log_performance(self, metrics):
        self.logger.info(f"PERFORMANCE: {metrics}")
