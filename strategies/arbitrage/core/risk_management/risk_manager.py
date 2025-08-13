#!/usr/bin/env python3
"""
Gestionnaire de risques central pour l'arbitrage
Créé: 2025-05-23
"""
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal

# Importation des composants de gestion des risques
from .position_sizer import PositionSizer
from .stop_loss import StopLossManager
from .exposure_limiter import ExposureLimiter

# Configuration du logging
logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gestionnaire central des risques pour les stratégies d'arbitrage.
    Intègre plusieurs composants de gestion des risques:
    - Dimensionnement des positions
    - Gestion des stop loss
    - Limites d'exposition
    - Protection contre le drawdown
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialisation du gestionnaire de risques
        Args:
            config_file: Chemin vers le fichier de configuration, utilise les valeurs par défaut si None
        """
        # Charger la configuration
        self.config = self._load_config(config_file)
        # Initialiser les composants
        self.position_sizer = PositionSizer(
            max_position=self.config["position_sizing"]["max_position"],
            max_percentage=self.config["position_sizing"]["max_percentage"],
            initial_size=self.config["position_sizing"]["initial_size"],
        )
        self.stop_loss = StopLossManager(
            global_stop_loss=self.config["stop_loss"]["global_stop_loss"],
            per_trade_stop_loss=self.config["stop_loss"]["per_trade_stop_loss"],
            trailing_stop=self.config["stop_loss"]["trailing_stop"],
        )
        self.exposure_limiter = ExposureLimiter(
            max_exposure=self.config["exposure_limits"]["max_exposure"],
            max_trades=self.config["exposure_limits"]["max_trades"],
            max_concurrent_exchanges=self.config["exposure_limits"][
                "max_concurrent_exchanges"
            ],
        )
        # État interne
        self.active_trades = {}
        self.trade_history = []
        self.current_exposure = Decimal("0")
        self.peak_balance = Decimal("0")
        self.last_update = 0

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Charge la configuration du gestionnaire de risques
        Args:
            config_file: Chemin vers le fichier de configuration
        Returns:
            Dictionnaire de configuration
        """
        # Configuration par défaut
        default_config = {
            "position_sizing": {
                "max_position": 1000.0,  # Taille maximale de position en USD
                "max_percentage": 5.0,  # Pourcentage maximal du capital par trade
                "initial_size": 100.0,  # Taille initiale de position en USD
            },
            "stop_loss": {
                "global_stop_loss": 10.0,  # Stop loss global en pourcentage
                "per_trade_stop_loss": 2.0,  # Stop loss par trade en pourcentage
                "trailing_stop": 1.0,  # Stop trailing en pourcentage
            },
            "exposure_limits": {
                "max_exposure": 50.0,  # Exposition maximale en pourcentage du capital
                "max_trades": 5,  # Nombre maximal de trades simultanés
                "max_concurrent_exchanges": 3,  # Nombre maximal d'exchanges utilisés simultanément
            },
            "drawdown_protection": {
                "max_drawdown": 15.0,  # Drawdown maximal autorisé en pourcentage
                "cooling_period": 3600,  # Période de refroidissement en secondes (1h)
                "reduction_factor": 0.5,  # Facteur de réduction de l'exposition après drawdown
            },
        }
        # Si aucun fichier de configuration n'est spécifié, utiliser les valeurs par défaut
        if config_file is None:
            return default_config
        # Charger la configuration depuis le fichier
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            # Fusionner avec les valeurs par défaut pour les clés manquantes
            for section, values in default_config.items():
                if section not in config:
                    config[section] = values
                else:
                    for key, value in values.items():
                        if key not in config[section]:
                            config[section][key] = value
            logger.info(f"Configuration chargée depuis {config_file}")
            return config
        except Exception as e:
            logger.warning(
                f"Erreur lors du chargement de la configuration: {e}, utilisation des valeurs par défaut"
            )
            return default_config

    def save_config(self, config_file: str) -> bool:
        """
        Sauvegarde la configuration actuelle dans un fichier
        Args:
            config_file: Chemin vers le fichier de configuration
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            # Sauvegarder la configuration
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration sauvegardée dans {config_file}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False

    def update_balance(self, balance: float) -> None:
        """
        Met à jour le solde du compte et recalcule les limites
        Args:
            balance: Solde actuel du compte en USD
        """
        balance_decimal = Decimal(str(balance))
        # Mettre à jour le pic de solde si nécessaire
        if balance_decimal > self.peak_balance:
            self.peak_balance = balance_decimal
        # Mettre à jour les composants
        self.position_sizer.update_account_balance(balance)
        self.exposure_limiter.update_account_balance(balance)
        # Vérifier le drawdown
        self._check_drawdown(balance_decimal)
        # Mettre à jour le timestamp
        self.last_update = time.time()

    def _check_drawdown(self, current_balance: Decimal) -> None:
        """
        Vérifie si le drawdown a dépassé la limite et applique les mesures de protection
        Args:
            current_balance: Solde actuel du compte
        """
        # Éviter la division par zéro
        if self.peak_balance == Decimal("0"):
            return
        # Calculer le drawdown en pourcentage
        drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
        # Vérifier si le drawdown dépasse la limite
        max_drawdown = Decimal(str(self.config["drawdown_protection"]["max_drawdown"]))
        if drawdown > max_drawdown:
            logger.warning(
                f"Drawdown critique détecté: {drawdown:.2f}% > {max_drawdown:.2f}%"
            )
            # Appliquer les mesures de protection
            reduction_factor = Decimal(
                str(self.config["drawdown_protection"]["reduction_factor"])
            )
            # Réduire les limites de position et d'exposition
            self.position_sizer.reduce_limits(reduction_factor)
            self.exposure_limiter.reduce_limits(reduction_factor)
            # Loguer l'événement
            logger.info(
                f"Limites de trading réduites par un facteur de {reduction_factor} en raison du drawdown"
            )

    def evaluate_arbitrage_opportunity(
        self, opportunity: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Évalue une opportunité d'arbitrage selon les critères de risque
        Args:
            opportunity: Dictionnaire décrivant l'opportunité d'arbitrage
        Returns:
            Tuple (approved, modified_opportunity)
            - approved: True si l'opportunité est approuvée, False sinon
            - modified_opportunity: Opportunité modifiée selon les contraintes de risque
        """
        # Extraire les informations de l'opportunité
        exchange_a = opportunity.get("exchange_a", "unknown")
        exchange_b = opportunity.get("exchange_b", "unknown")
        pair = opportunity.get("pair", "unknown")
        spread = Decimal(str(opportunity.get("spread", 0)))
        volume = Decimal(str(opportunity.get("volume", 0)))
        timestamp = opportunity.get("timestamp", time.time())
        # Vérifier les limites d'exposition
        if not self.exposure_limiter.can_take_new_trade(exchange_a, exchange_b):
            logger.warning(f"Opportunité refusée: limites d'exposition atteintes")
            return False, opportunity
        # Calculer la taille de position optimale
        suggested_size = self.position_sizer.calculate_position_size(spread, pair)
        # Limiter par le volume disponible
        position_size = min(suggested_size, volume)
        # Vérifier les stop loss
        if not self.stop_loss.is_trade_allowed(pair, exchange_a, exchange_b):
            logger.warning(
                f"Opportunité refusée: stop loss activé pour {pair} sur {exchange_a}/{exchange_b}"
            )
            return False, opportunity
        # Vérifier le cooling period après un drawdown
        cooling_period = self.config["drawdown_protection"]["cooling_period"]
        if time.time() - self.last_update < cooling_period:
            logger.info(
                f"En période de refroidissement: {cooling_period - (time.time() - self.last_update):.0f}s restantes"
            )
            # Réduire davantage la taille de position pendant la période de refroidissement
            position_size = position_size * Decimal("0.5")
        # Mettre à jour l'opportunité avec la taille de position calculée
        modified_opportunity = opportunity.copy()
        modified_opportunity["position_size"] = float(position_size)
        modified_opportunity["original_volume"] = float(volume)
        modified_opportunity["risk_adjusted"] = True
        # Approuver l'opportunité
        return True, modified_opportunity

    def register_trade(self, trade: Dict[str, Any]) -> str:
        """
        Enregistre un nouveau trade et met à jour l'état de gestion des risques
        Args:
            trade: Dictionnaire décrivant le trade
        Returns:
            ID du trade
        """
        # Générer un ID unique pour le trade
        trade_id = f"{int(time.time())}-{len(self.trade_history)}"
        # Ajouter l'ID au trade
        trade["id"] = trade_id
        # Extraire les informations du trade
        exchange_a = trade.get("exchange_a", "unknown")
        exchange_b = trade.get("exchange_b", "unknown")
        pair = trade.get("pair", "unknown")
        amount = Decimal(str(trade.get("amount", 0)))
        price = Decimal(str(trade.get("price", 0)))
        # Calculer la valeur du trade
        trade_value = amount * price
        # Mettre à jour l'exposition
        self.current_exposure += trade_value
        # Enregistrer le trade
        self.active_trades[trade_id] = trade
        # Informer les composants
        self.exposure_limiter.register_trade(
            trade_id, exchange_a, exchange_b, trade_value
        )
        self.stop_loss.register_trade(
            trade_id, pair, exchange_a, exchange_b, trade_value
        )
        logger.info(
            f"Trade enregistré: {trade_id} - {pair} sur {exchange_a}/{exchange_b} pour {trade_value:.2f} USD"
        )
        return trade_id

    def close_trade(
        self, trade_id: str, profit_loss: float, notes: Optional[str] = None
    ) -> bool:
        """
        Clôture un trade et met à jour l'état de gestion des risques
        Args:
            trade_id: ID du trade à clôturer
            profit_loss: Profit ou perte en USD
            notes: Notes supplémentaires sur la clôture du trade
        Returns:
            True si le trade a été clôturé, False si le trade n'existe pas
        """
        # Vérifier si le trade existe
        if trade_id not in self.active_trades:
            logger.warning(f"Tentative de clôture d'un trade inexistant: {trade_id}")
            return False
        # Récupérer le trade
        trade = self.active_trades[trade_id]
        # Extraire les informations du trade
        exchange_a = trade.get("exchange_a", "unknown")
        exchange_b = trade.get("exchange_b", "unknown")
        pair = trade.get("pair", "unknown")
        trade_value = Decimal(str(trade.get("amount", 0))) * Decimal(
            str(trade.get("price", 0))
        )
        # Mettre à jour l'exposition
        self.current_exposure -= trade_value
        # Ajouter les informations de clôture
        trade["closed_at"] = time.time()
        trade["profit_loss"] = profit_loss
        trade["notes"] = notes
        # Déplacer le trade vers l'historique
        self.trade_history.append(trade)
        del self.active_trades[trade_id]
        # Informer les composants
        self.exposure_limiter.close_trade(trade_id)
        self.stop_loss.update_trade_result(trade_id, profit_loss)
        # Ajuster les paramètres en fonction du résultat
        pl_decimal = Decimal(str(profit_loss))
        if pl_decimal > Decimal("0"):
            # Trade profitable, potentiellement augmenter les limites
            if len(self.trade_history) % 5 == 0:  # Tous les 5 trades profitables
                self.position_sizer.increase_limits(Decimal("1.05"))  # +5%
        elif pl_decimal < Decimal("-10"):  # Perte significative
            # Réduire les limites temporairement
            self.position_sizer.reduce_limits(Decimal("0.9"))  # -10%
        logger.info(
            f"Trade clôturé: {trade_id} - {pair} sur {exchange_a}/{exchange_b} avec P/L de {profit_loss:.2f} USD"
        )
        return True

    def get_risk_status(self) -> Dict[str, Any]:
        """
        Récupère l'état actuel de la gestion des risques
        Returns:
            Dictionnaire avec l'état de la gestion des risques
        """
        # Calculer le drawdown actuel
        drawdown = 0
        if self.peak_balance > Decimal("0"):
            current_balance = self.position_sizer.account_balance
            drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
        return {
            "active_trades": len(self.active_trades),
            "trade_history_count": len(self.trade_history),
            "current_exposure": float(self.current_exposure),
            "exposure_percentage": (
                float(self.current_exposure / self.position_sizer.account_balance * 100)
                if self.position_sizer.account_balance > 0
                else 0
            ),
            "max_exposure": float(self.exposure_limiter.max_exposure),
            "peak_balance": float(self.peak_balance),
            "current_balance": float(self.position_sizer.account_balance),
            "drawdown": float(drawdown),
            "position_sizing": {
                "current_max_position": float(self.position_sizer.max_position),
                "current_max_percentage": float(self.position_sizer.max_percentage),
            },
            "stop_loss": self.stop_loss.get_status(),
            "exposure_limits": self.exposure_limiter.get_status(),
        }

    def force_stop_trading(self) -> None:
        """
        Force l'arrêt du trading en cas d'urgence
        """
        logger.warning("ARRÊT D'URGENCE DU TRADING DÉCLENCHÉ")
        # Mettre les limites à zéro
        self.position_sizer.max_position = Decimal("0")
        self.exposure_limiter.max_exposure = Decimal("0")
        # Activer les stop loss globaux
        self.stop_loss.force_global_stop()

    def reset_risk_limits(self) -> None:
        """
        Réinitialise les limites de risque aux valeurs de la configuration
        """
        logger.info("Réinitialisation des limites de risque")
        # Réinitialiser les composants
        self.position_sizer = PositionSizer(
            max_position=self.config["position_sizing"]["max_position"],
            max_percentage=self.config["position_sizing"]["max_percentage"],
            initial_size=self.config["position_sizing"]["initial_size"],
        )
        self.stop_loss = StopLossManager(
            global_stop_loss=self.config["stop_loss"]["global_stop_loss"],
            per_trade_stop_loss=self.config["stop_loss"]["per_trade_stop_loss"],
            trailing_stop=self.config["stop_loss"]["trailing_stop"],
        )
        self.exposure_limiter = ExposureLimiter(
            max_exposure=self.config["exposure_limits"]["max_exposure"],
            max_trades=self.config["exposure_limits"]["max_trades"],
            max_concurrent_exchanges=self.config["exposure_limits"][
                "max_concurrent_exchanges"
            ],
        )
        # Mettre à jour avec le solde actuel
        balance = self.position_sizer.account_balance
        self.update_balance(float(balance))


# Test direct du module
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Initialiser le gestionnaire de risques
    risk_manager = RiskManager()
    # Définir un solde initial
    risk_manager.update_balance(10000.0)
    # Exemple d'opportunité d'arbitrage
    opportunity = {
        "exchange_a": "binance",
        "exchange_b": "gateio",
        "pair": "BTC/USDT",
        "spread": 0.8,
        "volume": 500.0,
        "timestamp": time.time(),
    }
    # Évaluer l'opportunité
    approved, modified_opp = risk_manager.evaluate_arbitrage_opportunity(opportunity)
    if approved:
        print(
            f"Opportunité approuvée avec taille de position: {modified_opp['position_size']}"
        )
        # Enregistrer un trade fictif
        trade = {
            "exchange_a": modified_opp["exchange_a"],
            "exchange_b": modified_opp["exchange_b"],
            "pair": modified_opp["pair"],
            "amount": modified_opp["position_size"] / 50000,  # Prix BTC fictif
            "price": 50000,
            "timestamp": time.time(),
        }
        trade_id = risk_manager.register_trade(trade)
        # Afficher l'état de la gestion des risques
        print(json.dumps(risk_manager.get_risk_status(), indent=2))
        # Simuler la clôture du trade avec profit
        risk_manager.close_trade(trade_id, 25.0, "Test de clôture profitable")
        # Afficher l'état mis à jour
        print(json.dumps(risk_manager.get_risk_status(), indent=2))
    else:
        print("Opportunité refusée en raison des contraintes de risque")
