import time
import sys
import os
from pathlib import Path
# Ajout du répertoire racine au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.notifications import send_arbitrage_alert
from strategies.arbitrage.multi_exchange import MultiExchangeArbitrage
def main():
    print("=== Arbitrage Bot ===")
    arb = MultiExchangeArbitrage()
    try:
        while True:
            opportunity = arb.get_best_spread()
            if opportunity and opportunity.get('spread', 0) > 0.3:
                send_arbitrage_alert(opportunity)
            time.sleep(15)
    except KeyboardInterrupt:
        print("\nBot arrêté proprement")
if __name__ == "__main__":
    main()
