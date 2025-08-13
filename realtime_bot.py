import logging
import json  # Standard library first
import asyncio
import websockets
from pandas import DataFrame, Series, read_csv, to_numeric
from typing import Dict, List
from src.core_merged.technical_engine import TechnicalEngine
# ============ NOUVELLE FONCTIONNALITÉ ============ #
def safe_log(message: str, level: str = "info"):
    """Logging sécurisé avec format lazy"""
    if level == "info":
        logging.info("%s", message)
    elif level == "warning":
        logging.warning("%s", message)
    elif level == "error":
        logging.error("%s", message)
# ============ FIN AJOUT ============ #
# Configuration du logging existante (conservée)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_bot.log'),
        logging.StreamHandler()
    ]
)
class RealTimeBot:
    def __init__(self):
        self.tech_engine = TechnicalEngine()
        self.data_window = DataFrame(
            columns=['open', 'high', 'low', 'close', 'volume'])
        self.window_size = 100
    def _update_data_window(self, new_row: dict):
        """Met à jour la fenêtre de données"""
        new_df = DataFrame([new_row])
        if self.data_window.empty:
            self.data_window = new_df
        else:
            self.data_window = concat(
                [self.data_window, new_df],
                ignore_index=True
            ).tail(self.window_size)
    async def handle_socket(self):
        uri = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        async with websockets.connect(uri) as websocket:
            safe_log("Connecté au flux temps réel Binance")
            while True:
                try:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    kline = data['k']
                    self._update_data_window({
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v'])
                    })
                    if len(self.data_window) >= 20:
                        analysis = self.tech_engine.compute(self.data_window)
                        safe_log(f"Prix: {self.data_window['close'].iloc[-1]:.2f}")
                        if 'rsi' in analysis.get('momentum', {}):
                            safe_log(f"RSI: {analysis['momentum']['rsi'].iloc[-1]:.2f}")
                except json.JSONDecodeError as e:
                    safe_log(f"Erreur de décodage JSON: {str(e)}", "warning")
                except Exception as e:
                    safe_log(f"Erreur inattendue: {str(e)}", "error")
# ... (le reste de votre code original conservé tel quel) ...
if __name__ == '__main__':
    try:
        bot = RealTimeBot()
        asyncio.get_event_loop().run_until_complete(bot.handle_socket())
    except KeyboardInterrupt:
        safe_log("Arrêt manuel du bot")
    except Exception as e:
        safe_log(f"Erreur critique: {str(e)}", "error")
