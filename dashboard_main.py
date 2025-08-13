import asyncio
import logging
from datetime import datetime
from visualization.dashboard import TradingDashboard
async def main():
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('TradingBot')
    try:
        # Initialize dashboard with custom settings
        dashboard = TradingDashboard(
            update_interval=1000,  # 1 second updates
            max_history=50000,     # Store more historical data
            log_level='INFO'
        )
        # Start market data stream in background
        asyncio.create_task(dashboard.start_data_stream())
        # Run dashboard
        dashboard.run(
            host='0.0.0.0',  # Allow external access
            port=8050,       # Default Dash port
            debug=False      # Production mode
        )
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        raise
    finally:
        logger.info("Trading Dashboard shutdown complete")
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {e}")
