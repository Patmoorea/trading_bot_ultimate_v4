import os
import asyncio
import websockets
import json
from collections import deque, defaultdict
import pandas as pd
import lz4.frame
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from binance.client import Client


class BufferedWSCollector:
    def __init__(self, symbols, timeframes, maxlen=1000):
        """
        symbols: liste de symboles Binance ex: ['btcusdt', 'ethusdt']
        timeframes: liste de timeframes ex: ['1m', '5m']
        maxlen: taille du buffer circulaire (par symbole/timeframe)
        """
        self.symbols = [s.lower() for s in symbols]  # <-- POUR LE WS SEUL
        self.symbols_upper = [s.upper() for s in symbols]  # POUR LES ACCÈS BUFFER
        self.timeframes = timeframes
        self.maxlen = maxlen
        # Buffer RAM circulaire : dict[(symbol,tf)] -> deque
        self.buffers = defaultdict(lambda: deque(maxlen=maxlen))
        self.ws_tasks = []
        self.running = False

    def get_orderbook(self, symbol, limit=5):
        try:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            client = Client(api_key, api_secret)
            # CORRECTION: always remove any "/" from symbol for API call
            symbol_binance = symbol.replace("/", "").upper()
            ob = client.get_order_book(symbol=symbol_binance, limit=limit)
            best_bid = float(ob["bids"][0][0]) if ob["bids"] else None
            best_ask = float(ob["asks"][0][0]) if ob["asks"] else None
            return best_bid, best_ask
        except Exception as e:
            print(f"[WSCollector] Erreur récupération orderbook Binance : {e}")
            return None, None

    def _make_stream_url(self):
        streams = [
            f"{sym}@kline_{tf}" for sym in self.symbols for tf in self.timeframes
        ]
        stream_str = "/".join(streams)
        return f"wss://stream.binance.com:9443/stream?streams={stream_str}"

    async def _ws_loop(self):
        url = self._make_stream_url()
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    print(f"[WS] Connecté à {url}")
                    async for msg in ws:
                        data = json.loads(msg)
                        stream = data.get("stream", "")
                        if stream and "kline" in stream:
                            sym, tf = stream.split("@")[0], stream.split("_")[-1]
                            k = data["data"]["k"]
                            # print(f"[WS DEBUG] Reçu tick {sym} {tf}")
                            self._update_buffer(sym, tf, k)
            except Exception as e:
                print(f"[WS] Erreur {e}, reconnexion dans 5s")
                await asyncio.sleep(5)

    def update_incremental(self, new_data):
        for symbol, tf_dict in new_data.items():
            for tf, ohlcv in tf_dict.items():
                self.buffers[symbol][tf].extend(ohlcv)
                self.buffers[symbol][tf] = self.buffers[symbol][tf][-self.maxlen :]

    def _update_buffer(self, symbol, timeframe, kline):
        timestamp = datetime.fromtimestamp(kline["t"] / 1000)
        buf = self.buffers[(symbol.upper(), timeframe)]
        # Anti-doublon : n'ajoute pas si timestamp déjà dans le buffer
        if any(e["timestamp"] == timestamp for e in buf):
            return
        entry = {
            "timestamp": timestamp,
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "is_closed": bool(kline["x"]),
        }
        buf.append(entry)

    def get_dataframe(self, symbol, timeframe):
        buf = self.buffers.get((symbol.upper(), timeframe), [])
        df = pd.DataFrame(list(buf)) if buf else pd.DataFrame()
        if not df.empty and "timestamp" in df.columns:
            print(
                f"[DEBUG] Timestamps avant tri {symbol}-{timeframe}:",
                df["timestamp"].head(10).tolist(),
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.drop_duplicates(subset="timestamp", keep="last")
            df = df.sort_values("timestamp")
            df = df.reset_index(drop=True)
            df = df[df["timestamp"].diff().dt.total_seconds().fillna(1) > 0]
            if not df["timestamp"].is_monotonic_increasing:
                print(
                    f"[ERROR] Timestamps NOT strictly increasing in {symbol}-{timeframe}, DataFrame invalid!"
                )
                return pd.DataFrame()
        return df

    def get_last_price(self, symbol, timeframe="1h"):
        """
        Retourne le dernier prix close pour un symbole et timeframe donné (ex: BTCUSDC, "1h").
        """
        df = self.get_dataframe(symbol, timeframe)
        if not df.empty and "close" in df.columns:
            return float(df["close"].iloc[-1])
        return None

    def compress_buffer(self, symbol, timeframe):
        df = self.get_dataframe(symbol, timeframe)
        if df.empty:
            return None
        bin_data = df.to_parquet(index=False)
        compressed = lz4.frame.compress(bin_data)
        return compressed

    def save_parquet(self, symbol, timeframe, path):
        df = self.get_dataframe(symbol, timeframe)
        if not df.empty:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)

    def preload_historical(self, client, symbol, timeframe, limit=1000):
        tf_dict = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }
        # CORRECTION: always remove "/" from symbol for API call
        symbol_binance = symbol.replace("/", "").upper()
        klines = client.get_klines(
            symbol=symbol_binance, interval=tf_dict[timeframe], limit=limit
        )
        for k in klines:
            entry = {
                "timestamp": datetime.fromtimestamp(k[0] / 1000),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "is_closed": True,
            }
            buf = self.buffers[(symbol_binance, timeframe)]
            if any(e["timestamp"] == entry["timestamp"] for e in buf):
                continue
            buf.append(entry)

    async def start(self):
        self.running = True
        self.ws_tasks.append(asyncio.create_task(self._ws_loop()))

    async def stop(self):
        self.running = False
        for t in self.ws_tasks:
            t.cancel()
        self.ws_tasks.clear()


# Exemple d'intégration autonome
if __name__ == "__main__":

    async def main():
        collector = BufferedWSCollector(
            symbols=["BTCUSDT", "ETHUSDT"], timeframes=["1m", "5m"], maxlen=500
        )
        await collector.start()
        await asyncio.sleep(120)  # Reste connecté 2min
        await collector.stop()
        # Sauvegarde exemple avec la bonne casse (UPPER)
        for sym in collector.symbols_upper:
            for tf in collector.timeframes:
                collector.save_parquet(sym, tf, f"{sym}_{tf}.parquet")
                compressed = collector.compress_buffer(sym, tf)
                if compressed:
                    with open(f"{sym}_{tf}.lz4", "wb") as f:
                        f.write(compressed)
                # Ajoute un debug pour voir la taille du DataFrame
                df = collector.get_dataframe(sym, tf)
                print(f"[DEBUG] {sym} {tf}: {len(df)} lignes")

    asyncio.run(main())
