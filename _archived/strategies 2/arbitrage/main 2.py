import asyncio
import time
from decimal import Decimal
from utils.logger import get_logger
from .service import MoteurArbitrage
from .config import SETTINGS
logger = get_logger()
class BotArbitrage:
    def __init__(self):
        self.moteur = MoteurArbitrage()
        self.profit_total = Decimal(0)
        self.debut = time.time()
    async def executer(self):
        logger.info("Démarrage du bot d'arbitrage...")
        while True:
            debut_iteration = time.time()
            try:
                opportunites = await self.moteur.scanner_opportunites()
                if not opportunites:
                    logger.info("Aucune opportunité d'arbitrage trouvée")
                else:
                    for opp in sorted(opportunites, key=lambda x: x.profit, reverse=True):
                        try:
                            profit = await self.moteur.executer_arbitrage(opp)
                            self.profit_total += Decimal(str(profit))
                            logger.info(f"Profit total: {self.profit_total:.2f} USD | "
                                      f"Durée: {(time.time() - self.debut)/60:.1f} minutes")
                        except Exception as e:
                            logger.error(f"Échec exécution arbitrage: {str(e)}")
                            continue
                        await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Erreur boucle principale: {str(e)}")
            ecoule = time.time() - debut_iteration
            attente = max(0, SETTINGS['price_expiry'] - ecoule)
            await asyncio.sleep(attente)
if __name__ == "__main__":
    bot = BotArbitrage()
    try:
        asyncio.run(bot.executer())
    except KeyboardInterrupt:
        logger.info("Bot arrêté par l'utilisateur")
