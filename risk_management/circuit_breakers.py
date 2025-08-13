"""
Circuit Breakers Module
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
logger = logging.getLogger(__name__)
class CircuitBreaker:
    def __init__(
        self,
        crash_threshold: float = 0.1,
        liquidity_threshold: float = 0.5,
        volatility_threshold: float = 0.3,
        cooldown_period: int = 300,  # 5 minutes en secondes
        max_triggers_per_day: int = 3
    ):
        """
        Initialise le circuit breaker
        :param crash_threshold: Seuil de chute de prix pour activation (10% par défaut)
        :param liquidity_threshold: Seuil de liquidité minimum (50% par défaut)
        :param volatility_threshold: Seuil de volatilité maximum (30% par défaut)
        :param cooldown_period: Période de refroidissement en secondes
        :param max_triggers_per_day: Nombre maximum d'activations par jour
        """
        self.crash_threshold = crash_threshold
        self.liquidity_threshold = liquidity_threshold
        self.volatility_threshold = volatility_threshold
        self.cooldown_period = cooldown_period
        self.max_triggers_per_day = max_triggers_per_day
        self.is_active = False
        self.last_trigger_time = None
        self.trigger_count = 0
        self.last_trigger_reason = None
        self.triggers_history = []
    async def should_stop_trading(self) -> bool:
        """Vérifie si le trading doit être arrêté"""
        try:
            # Si déjà actif, vérifier la période de refroidissement
            if self.is_active:
                if self.last_trigger_time:
                    elapsed = (datetime.utcnow() - self.last_trigger_time).total_seconds()
                    if elapsed < self.cooldown_period:
                        return True
                    self.is_active = False
                    self.last_trigger_reason = None
                return False
            # Vérification du nombre de triggers journaliers
            today_triggers = sum(1 for t in self.triggers_history 
                              if (datetime.utcnow() - t["timestamp"]).days < 1)
            if today_triggers >= self.max_triggers_per_day:
                logger.warning(f"[{self.current_time}] Nombre maximum de triggers journaliers atteint")
                return True
            return False
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur vérification circuit breaker: {e}")
            return True  # Par sécurité
    async def check_market_crash(self, price_change: float) -> bool:
        """Vérifie si une chute de marché est en cours"""
        try:
            if abs(price_change) > self.crash_threshold:
                self._trigger_breaker("MARKET_CRASH")
                logger.warning(f"[{self.current_time}] Circuit breaker activé: Chute de marché ({price_change:.2%})")
                return True
            return False
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur vérification crash: {e}")
            return True
    async def check_liquidity_shock(self, liquidity_ratio: float) -> bool:
        """Vérifie si un choc de liquidité est en cours"""
        try:
            if liquidity_ratio < self.liquidity_threshold:
                self._trigger_breaker("LIQUIDITY_SHOCK")
                logger.warning(f"[{self.current_time}] Circuit breaker activé: Choc de liquidité ({liquidity_ratio:.2%})")
                return True
            return False
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur vérification liquidité: {e}")
            return True
    async def check_volatility_spike(self, volatility: float) -> bool:
        """Vérifie si un pic de volatilité est en cours"""
        try:
            if volatility > self.volatility_threshold:
                self._trigger_breaker("VOLATILITY_SPIKE")
                logger.warning(f"[{self.current_time}] Circuit breaker activé: Pic de volatilité ({volatility:.2%})")
                return True
            return False
        except Exception as e:
            logger.error(f"[{self.current_time}] Erreur vérification volatilité: {e}")
            return True
    def _trigger_breaker(self, reason: str) -> None:
        """Active le circuit breaker"""
        self.is_active = True
        self.last_trigger_time = datetime.utcnow()
        self.last_trigger_reason = reason
        self.trigger_count += 1
        # Enregistrement du trigger
        self.triggers_history.append({
            "timestamp": datetime.utcnow(),
            "reason": reason,
            "count": self.trigger_count
        })
        # Nettoyage de l'historique (garder uniquement la dernière semaine)
        week_ago = datetime.utcnow() - timedelta(days=7)
        self.triggers_history = [t for t in self.triggers_history 
                               if t["timestamp"] > week_ago]
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel du circuit breaker"""
        return {
            "is_active": self.is_active,
            "last_trigger_time": self.last_trigger_time,
            "last_trigger_reason": self.last_trigger_reason,
            "trigger_count": self.trigger_count,
            "daily_triggers": len([t for t in self.triggers_history 
                                 if (datetime.utcnow() - t["timestamp"]).days < 1]),
            "timestamp": self.current_time
        }
    async def reset(self) -> None:
        """Réinitialise le circuit breaker"""
        self.is_active = False
        self.last_trigger_time = None
        self.trigger_count = 0
        self.last_trigger_reason = None
        self.triggers_history.clear()
        logger.info(f"[{self.current_time}] Circuit breaker réinitialisé")
