import asyncio
import os
from dotenv import load_dotenv
import redis.asyncio as redis
from binance import AsyncClient, BinanceSocketManager
load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOL = "ltcusdt"
async def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    client = await AsyncClient.create(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    bm = BinanceSocketManager(client)
    ts = bm.trade_socket(SYMBOL)
    async with ts as tscm:
        while True:
            res = await tscm.recv()
            price = float(res['p'])
            event_time = res['E']
            await r.set(f"{SYMBOL}_last_price", price)
            await r.set(f"{SYMBOL}_last_event_time", event_time)
            print(f"Prix {SYMBOL.upper()} mis à jour: {price}")
    await client.close_connection()
if __name__ == "__main__":
    asyncio.run(main())
