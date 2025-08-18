from binance.client import Client
import pandas as pd

# 1. Mets tes clés ici
api_key = "NnbmLwP1glwkXqq5JWVMjoCnc4dnCewlyeY3v7dgXkx7aoLopWpbM6rhEZyFMlJt"
api_secret = "vby8jHesZZpYYeEcmauzJqx4PKvUOYXXZVAxgzfDqNnTUWXbqi0L6Ia8abHnvc7T"

# 2. Crée le client Binance
client = Client(api_key, api_secret)

# 3. Télécharge les données historiques
klines = client.get_historical_klines(
    "BTCUSDC",  # Change la paire si tu veux
    Client.KLINE_INTERVAL_1HOUR,
    "1 Jan, 2023",  # Date de début
    "now"           # Date de fin
)

# 4. Convertis en DataFrame
df = pd.DataFrame(klines, columns=[
    "timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
    "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
])
df = df[["timestamp", "open", "high", "low", "close", "volume"]]
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# 5. Sauve en CSV
df.to_csv("BTCUSDC_1h.csv", index=False)
print("CSV généré : BTCUSDC_1h.csv")