# Ajout de la fonction send_arbitrage_alert pour le module arbitrage
from typing import Dict, Any
from decimal import Decimal
import logging
from datetime import datetime
logger = logging.getLogger(__name__)
async def send_arbitrage_alert(opportunity: Dict[str, Any]) -> None:
    """
    Envoie une alerte d'arbitrage
    Args:
        opportunity: Dictionnaire contenant les détails de l'opportunité
    """
    try:
        message = (
            f"🔄 Opportunité d'arbitrage [{current_time} UTC]\n\n"
            f"Spread: {Decimal(str(opportunity.get('spread', 0))) * 100:.2f}%\n"
            f"Exchange achat: {opportunity.get('buy_exchange', 'N/A')}\n" 
            f"Exchange vente: {opportunity.get('sell_exchange', 'N/A')}\n"
            f"Prix achat: {opportunity.get('buy_price', 0):.2f}\n"
            f"Prix vente: {opportunity.get('sell_price', 0):.2f}\n"
            f"Volume: {opportunity.get('volume', 0):.4f}\n"
            f"Profit potentiel: {opportunity.get('profit', 0):.2f} USDT"
        )
        logger.info(f"Alerte d'arbitrage envoyée: {message}")
        print(message)  # Version simple pour debug
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'alerte d'arbitrage: {e}")
        raise
__all__ = ['send_arbitrage_alert']
