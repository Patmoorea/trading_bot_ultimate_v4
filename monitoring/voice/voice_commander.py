import speech_recognition as sr
import torch
import numpy as np
from typing import Dict, Callable, Optional
import sounddevice as sd
import soundfile as sf
class VoiceCommander:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.commands = self._init_commands()
        self.alerts = self._load_alerts()
    def _init_commands(self) -> Dict[str, Callable]:
        return {
            "stop trading": self._stop_trading,
            "show portfolio": self._show_portfolio,
            "market status": self._market_status,
            "place order": self._place_order,
            "cancel orders": self._cancel_orders,
            "risk status": self._risk_status,
            "performance": self._show_performance,
            "alert settings": self._alert_settings,
            "system health": self._system_health,
            "help": self._show_help
        }
    def _load_alerts(self) -> Dict[str, str]:
        return {
            "stop_loss": "alerts/stop_loss.wav",
            "take_profit": "alerts/take_profit.wav",
            "margin_call": "alerts/margin_call.wav",
            "market_move": "alerts/market_move.wav",
            "system_error": "alerts/system_error.wav"
        }
    def listen(self) -> Optional[Dict]:
        """Écoute et traite les commandes vocales"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                return self._process_command(text.lower())
            except sr.UnknownValueError:
                return {"status": "error", "message": "Commande non reconnue"}
            except sr.RequestError:
                return {"status": "error", "message": "Service non disponible"}
    def play_alert(self, alert_type: str) -> None:
        """Joue une alerte sonore"""
        if alert_type in self.alerts:
            data, fs = sf.read(self.alerts[alert_type])
            sd.play(data, fs)
            sd.wait()
    def _process_command(self, text: str) -> Dict:
        """Traite une commande vocale"""
        for command, func in self.commands.items():
            if command in text:
                return func()
        return {"status": "error", "message": "Commande inconnue"}
    # Implémentation des commandes
    def _stop_trading(self) -> Dict:
        return {"action": "stop_trading", "executed": True}
    def _show_portfolio(self) -> Dict:
        return {"action": "show_portfolio", "executed": True}
    def _market_status(self) -> Dict:
        return {"action": "market_status", "executed": True}
    def _place_order(self) -> Dict:
        return {"action": "place_order", "executed": True}
    def _cancel_orders(self) -> Dict:
        return {"action": "cancel_orders", "executed": True}
    def _risk_status(self) -> Dict:
        return {"action": "risk_status", "executed": True}
    def _show_performance(self) -> Dict:
        return {"action": "show_performance", "executed": True}
    def _alert_settings(self) -> Dict:
        return {"action": "alert_settings", "executed": True}
    def _system_health(self) -> Dict:
        return {"action": "system_health", "executed": True}
    def _show_help(self) -> Dict:
        return {"action": "show_help", "executed": True}
