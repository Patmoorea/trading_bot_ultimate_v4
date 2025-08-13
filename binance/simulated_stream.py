import asyncio
import random
import time
async def simulated_price_stream():
    price = 50.0
    while True:
        price += random.uniform(-0.5, 0.5)
        price = max(price, 0.1)
        print(f"Prix simulé: {price:.2f} USDT")
        await asyncio.sleep(1)
if __name__ == "__main__":
    asyncio.run(simulated_price_stream())
