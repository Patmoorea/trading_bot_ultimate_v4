import aiohttp
import asyncio
import pandas as pd
from transformers import pipeline
from typing import List, Dict, Any
import logging
import feedparser
from dotenv import load_dotenv

load_dotenv()


class NewsAnalyzer:
    def __init__(self, cryptopanic_api_key: str = None):
        self.sources = [
            "cryptopanic.com",
            "coindesk.com",
            "cointelegraph.com",
            "bitcoinmagazine.com",
            "decrypt.co",
            "theblockcrypto.com",
            "newsbtc.com",
            "bitcoinist.com",
            "cryptoslate.com",
            "cryptobriefing.com",
            "ambcrypto.com",
            "beincrypto.com",
        ]
        self.sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        self.logger = logging.getLogger(__name__)
        self.cryptopanic_api_key = cryptopanic_api_key

        # Mapping RSS for each source
        self.rss_feeds = {
            "coindesk.com": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "bitcoinmagazine.com": "https://bitcoinmagazine.com/.rss/full/",
            "decrypt.co": "https://decrypt.co/feed",
            "theblockcrypto.com": "https://www.theblock.co/rss.xml",
            "newsbtc.com": "https://www.newsbtc.com/feed/",
            "bitcoinist.com": "https://bitcoinist.com/feed/",
            "cryptoslate.com": "https://cryptoslate.com/feed/",
            "cryptobriefing.com": "https://cryptobriefing.com/feed/",
            "ambcrypto.com": "https://ambcrypto.com/feed/",
            "beincrypto.com": "https://beincrypto.com/feed/",
        }

    async def fetch_news(self) -> list:
        # Timeout global pour la session (8s, mais chaque requête aura son propre timeout)
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Crée une tâche par source (fetch_source doit gérer son propre timeout)
            tasks = [self._fetch_source(session, source) for source in self.sources]
            # Attend au max 10s pour toutes les tâches, puis annule les trop lentes
            done, pending = await asyncio.wait(tasks, timeout=10)
            news_items = []
            for task in done:
                try:
                    result = task.result()
                    news_items.extend(result)
                except Exception as e:
                    self.logger.error(f"Erreur récupération news: {str(e)}")
            for task in pending:
                task.cancel()
                self.logger.warning(f"Requête news annulée (trop longue)")
            return news_items

    async def _fetch_source(
        self, session: aiohttp.ClientSession, source: str
    ) -> List[Dict[str, Any]]:
        if "cointelegraph.com" in source:
            return await self._fetch_cointelegraph(session)
        elif "coindesk.com" in source:
            return await self._fetch_rss(session, "coindesk.com")
        elif "bitcoinmagazine.com" in source:
            return await self._fetch_rss(session, "bitcoinmagazine.com")
        elif "decrypt.co" in source:
            return await self._fetch_rss(session, "decrypt.co")
        elif "theblockcrypto.com" in source:
            return await self._fetch_rss(session, "theblockcrypto.com")
        elif "newsbtc.com" in source:
            return await self._fetch_rss(session, "newsbtc.com")
        elif "bitcoinist.com" in source:
            return await self._fetch_rss(session, "bitcoinist.com")
        elif "cryptoslate.com" in source:
            return await self._fetch_rss(session, "cryptoslate.com")
        elif "cryptobriefing.com" in source:
            return await self._fetch_rss(session, "cryptobriefing.com")
        elif "ambcrypto.com" in source:
            return await self._fetch_rss(session, "ambcrypto.com")
        elif "beincrypto.com" in source:
            return await self._fetch_rss(session, "beincrypto.com")
        elif "cryptopanic.com" in source:
            return await self._fetch_cryptopanic(session)
        else:
            self.logger.warning(f"No handler for source: {source}")
            return []

    async def _fetch_cointelegraph(self, session):
        url = "https://cointelegraph.com/api/v1/news"
        headers = {"User-Agent": "Mozilla/5.0"}
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200 and resp.headers.get("Content-Type", "").startswith(
                "application/json"
            ):
                data = await resp.json()
                return [
                    {
                        "source": "cointelegraph.com",
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "url": item.get("url", ""),
                    }
                    for item in data.get("data", [])
                ]
            else:
                self.logger.error(
                    f"Erreur Cointelegraph: code={resp.status}, content-type={resp.headers.get('Content-Type')}"
                )
                return []

    async def _fetch_rss(self, session, source_key):
        url = self.rss_feeds[source_key]
        try:
            # Timeout individuel ici : 6 secondes max
            async with session.get(url, timeout=6) as resp:
                if resp.status != 200:
                    self.logger.error(f"Erreur RSS {source_key}: code={resp.status}")
                    return []
                content = await resp.read()
                ...
        except asyncio.TimeoutError:
            self.logger.warning(f"[{source_key}] Timeout après 6s")
            return []
        except Exception as e:
            self.logger.error(f"Erreur parsing RSS {source_key}: {e}")
            return []

        async def _fetch_cryptopanic(self, session):
            """
            Récupère les news via l'API Cryptopanic avec la clé dans .env (CRYPTOPANIC_API_KEY).
            """
            # Charge la clé API depuis .env (pense à: from dotenv import load_dotenv; load_dotenv() au démarrage)
            api_key = os.getenv("CRYPTOPANIC_API_KEY")
            if not api_key:
                self.logger.error(
                    "Clé API Cryptopanic manquante dans .env (CRYPTOPANIC_API_KEY)"
                )
                return []

        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&public=true"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = []
                    for item in data.get("results", []):
                        results.append(
                            {
                                "source": "cryptopanic.com",
                                "title": item.get("title", ""),
                                "description": item.get("description", ""),
                                "url": item.get("url", ""),
                                "published": item.get("published_at", ""),
                                "currencies": (
                                    [c["code"] for c in item.get("currencies", [])]
                                    if item.get("currencies")
                                    else []
                                ),
                                "domain": item.get("domain", ""),
                            }
                        )
                    return results
                else:
                    self.logger.error(f"Erreur Cryptopanic: code={resp.status}")
                    return []
        except Exception as e:
            self.logger.error(f"Erreur Cryptopanic: {e}")
            return []

    def analyze_sentiment(self, news_items: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyse le sentiment des news avec FinBERT"""
        texts = [item["title"] + " " + item["description"] for item in news_items]
        sentiments = self.sentiment_model(texts)
        df = pd.DataFrame(news_items)
        df["sentiment_score"] = [s["score"] for s in sentiments]
        df["sentiment"] = [s["label"] for s in sentiments]
        return df

    def calculate_market_impact(self, analyzed_news: pd.DataFrame) -> Dict[str, float]:
        """Calcule l'impact potentiel sur le marché"""
        impact = {"bullish_score": 0.0, "bearish_score": 0.0, "neutral_score": 0.0}
        # Pondération basée sur la source et le sentiment
        for _, news in analyzed_news.iterrows():
            weight = self._calculate_source_weight(news["source"])
            if news["sentiment"] == "positive":
                impact["bullish_score"] += news["sentiment_score"] * weight
            elif news["sentiment"] == "negative":
                impact["bearish_score"] += news["sentiment_score"] * weight
            else:
                impact["neutral_score"] += news["sentiment_score"] * weight
        return impact

    def _calculate_source_weight(self, source: str) -> float:
        """Calcule le poids de crédibilité d'une source"""
        weights = {
            "coindesk.com": 1.0,
            "cointelegraph.com": 0.9,
            "theblockcrypto.com": 0.95,
            "bitcoinmagazine.com": 0.85,
            "cryptopanic.com": 0.8,
            "decrypt.co": 0.85,
            "newsbtc.com": 0.8,
            "bitcoinist.com": 0.8,
            "cryptoslate.com": 0.8,
            "cryptobriefing.com": 0.8,
            "ambcrypto.com": 0.7,
            "beincrypto.com": 0.7,
        }
        return weights.get(source, 0.7)  # 0.7 par défaut
