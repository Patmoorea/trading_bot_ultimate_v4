import os
import re
import json
import logging
import aiohttp
import ssl
import asyncio
from typing import List, Dict, Optional, Set, Any
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from datetime import datetime, timezone
from deep_translator import GoogleTranslator


def save_shared_data(update_dict, data_file):
    """PATCH: Sauvegarde universelle du JSON racine, fusionne avec l'existant."""
    try:
        if os.path.exists(data_file):
            with open(data_file, "r") as f:
                shared_data = json.load(f)
                if not isinstance(shared_data, dict):
                    shared_data = {}
        else:
            shared_data = {}
        shared_data.update(update_dict)
        with open(data_file, "w") as f:
            json.dump(shared_data, f, indent=2)
    except Exception as e:
        print(f"[PATCH] Erreur sauvegarde JSON sentiment : {e}")


class NewsSentimentAnalyzer:
    SYMBOL_MAPPING = {
        "bitcoin": "BTC",
        "btc": "BTC",
        "ethereum": "ETH",
        "eth": "ETH",
        "cardano": "ADA",
        "ada": "ADA",
        "solana": "SOL",
        "sol": "SOL",
        "ripple": "XRP",
        "xrp": "XRP",
        "dogecoin": "DOGE",
        "doge": "DOGE",
        "polkadot": "DOT",
        "dot": "DOT",
        "binance": "BNB",
        "bnb": "BNB",
        "matic": "MATIC",
        "polygon": "MATIC",
        "litecoin": "LTC",
        "ltc": "LTC",
        "shiba": "SHIB",
        "shib": "SHIB",
        "tron": "TRX",
        "trx": "TRX",
        "avalanche": "AVAX",
        "avax": "AVAX",
        "chainlink": "LINK",
        "link": "LINK",
        "uniswap": "UNI",
        "uni": "UNI",
        "stellar": "XLM",
        "xlm": "XLM",
    }

    def real_translate_title(self, text):
        try:
            return self.translator.translate(text)
        except Exception:
            original = text
            dico = {
                "Bitcoin": "Bitcoin",
                "Ethereum": "Ethereum",
                "price": "prix",
                "update": "mise à jour",
                "reaches": "atteint",
                "falls": "chute",
                "surges": "explose",
                "network": "réseau",
                "record": "record",
                "launch": "lancement",
                "approval": "approbation",
                "hack": "piratage",
                "coin": "jeton",
                "exchange": "plateforme",
                "regulation": "réglementation",
                "ETF": "ETF",
                "market": "marché",
                "crash": "effondrement",
                "rise": "hausse",
                "buy": "achat",
                "sell": "vente",
                "token": "jeton",
                "trading": "trading",
                "volume": "volume",
                "support": "support",
                "resistance": "résistance",
            }
            for en, fr in dico.items():
                text = text.replace(en, fr)
            if text == original:
                try:
                    return self.translator.translate(text)
                except:
                    return text
            return text

    def __init__(self, config: dict):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.crypto_panic_api_key = os.getenv("CRYPTO_PANIC_API_KEY")
        self.news_api_languages = os.getenv("NEWS_API_LANGUAGES", "en,fr")
        self.news_sources = os.getenv("NEWS_SOURCES", "bloomberg,reuters,coindesk")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        self.conn_timeout = 5
        self.max_retries = 2
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
        self.regex_patterns = [
            (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), ticker)
            for name, ticker in self.SYMBOL_MAPPING.items()
        ]
        self.known_tickers = set(self.SYMBOL_MAPPING.values())

        watermark = config.get("news", {}).get("low_watermark_ratio", 0.75)
        try:
            watermark = float(watermark)
        except Exception:
            watermark = 0.75
        # PATCH: borne la valeur dans [0.05, 0.8]
        if watermark > 0.8 or watermark < 0.05:
            print(f"[PATCH] Watermark ratio {watermark} invalide, fallback à 0.75")
            watermark = 0.75
        self.low_watermark_ratio = watermark

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self._model = None
        self._tokenizer = None
        # LLM classifier pipeline
        self.llm_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # <-- AJOUTE CE PARAMÈTRE !
        )
        self.risk_labels = config.get("news", {}).get(
            "risk_labels",
            [
                "Regulatory",
                "Security",
                "Hack",
                "Scam",
                "Bullish",
                "Bearish",
                "Neutral",
                "Pump",
                "Crash",
            ],
        )
        self.news_buffer = []
        self.sentiment_weight = config.get("news", {}).get("sentiment_weight", 0.15)
        self.update_interval = config.get("news", {}).get("update_interval", 300)
        self.sources = [
            {
                "name": "CryptoCompare",
                "url": "https://min-api.cryptocompare.com/data/v2/news/?lang=FR",
                "type": "json",
                "weight": 0.7,
                "priority": 1,
            },
            {
                "name": "NewsAPI",
                "url": (
                    "https://newsapi.org/v2/everything?"
                    "q=crypto OR bitcoin OR blockchain&"
                    f"language={self.news_api_languages}&"
                    f"sources={self.news_sources}&"
                    f"apiKey={self.news_api_key}"
                ),
                "type": "json",
                "weight": 0.7,
                "priority": 1,
            },
            {
                "name": "Cointelegraph",
                "url": "https://cointelegraph.com/rss",
                "type": "rss",
                "weight": 0.8,
                "priority": 2,
            },
            {
                "name": "Decrypt",
                "url": "https://decrypt.co/feed",
                "type": "rss",
                "weight": 0.8,
                "priority": 2,
            },
            {
                "name": "NewsBTC",
                "url": "https://www.newsbtc.com/feed/",
                "type": "rss",
                "weight": 0.7,
                "priority": 2,
            },
            {
                "name": "TheBlock",
                "url": "https://www.theblock.co/rss.xml",
                "type": "rss",
                "weight": 0.7,
                "priority": 2,
            },
        ]
        self.translate_news = config.get("news", {}).get("translate", True)
        self.target_language = config.get("news", {}).get("language", "fr")
        self.translator = GoogleTranslator(source="auto", target=self.target_language)

    def _translate_text(self, text: str) -> str:
        try:
            if not text or not isinstance(text, str):
                return ""
            return self.translator.translate(text)
        except Exception as e:
            self.logger.warning(f"Erreur traduction: {str(e)}")
            return text

    def _parse_json(self, data, source: Dict) -> List[Dict]:
        news_list = []
        if source["name"] == "CryptoCompare" and "Data" in data:
            for n in data["Data"]:
                title = n.get("title", "")
                text = n.get("body", "")
                original_title = title
                if self.translate_news:
                    title = self.real_translate_title(title)
                    text = self.real_translate_title(text)
                url = n.get("url", "")
                symbols = self.extract_symbols(f"{title} {text}")
                timestamp = self.normalize_timestamp(n.get("published_on", None))
                news_list.append(
                    {
                        "title": title,
                        "text": text,
                        "original_title": original_title,
                        "source": source["name"],
                        "timestamp": timestamp,
                        "url": url,
                        "symbols": symbols,
                        "source_weight": source["weight"],
                        "processed": False,
                    }
                )
        elif source["name"] == "NewsAPI" and "articles" in data:
            for n in data["articles"]:
                title = n.get("title", "")
                text = n.get("description", "") or n.get("content", "")
                original_title = title
                if self.translate_news:
                    title = self.real_translate_title(title)
                    text = self.real_translate_title(text)
                url = n.get("url", "")
                symbols = self.extract_symbols(f"{title} {text}")
                timestamp = self.normalize_timestamp(n.get("publishedAt", None))
                news_list.append(
                    {
                        "title": title,
                        "text": text,
                        "original_title": original_title,
                        "source": source["name"],
                        "timestamp": timestamp,
                        "url": url,
                        "symbols": symbols,
                        "source_weight": source["weight"],
                        "processed": False,
                    }
                )
        return news_list

    def _parse_rss_item(self, item, source: Dict) -> Dict:
        title = item.find("title").text if item.find("title") else ""
        description = item.find("description").text if item.find("description") else ""
        original_title = title
        if self.translate_news:
            title = self.real_translate_title(title)
            description = self.real_translate_title(description)
        url = item.find("link").text if item.find("link") else ""
        symbols = self.extract_symbols(f"{title} {description}")
        pub_date = item.find("pubDate")
        timestamp = self.normalize_timestamp(pub_date.text if pub_date else None)
        return {
            "title": title,
            "text": description,
            "original_title": original_title,
            "source": source["name"],
            "timestamp": timestamp,
            "url": url,
            "symbols": symbols,
            "source_weight": source["weight"],
            "processed": False,
        }

    async def _save_state(self, data):
        path = self.config.get("news", {}).get(
            "storage_path", "data/news_analysis.json"
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_shared_data(data, path)
            self.logger.info(f"[NEWS] State saved to {path}")
        except Exception as e:
            self.logger.error(f"[NEWS] Failed to save state: {e}")

    def extract_symbols(self, text: str) -> List[str]:
        found: Set[str] = set()
        if not text:
            return []
        for pattern, ticker in self.regex_patterns:
            if pattern.search(text):
                found.add(ticker)
        for pair in re.findall(
            r"\b([A-Z]{3,5})[/-]?(USDT|USD|BTC|ETH)?\b", text.upper()
        ):
            ticker = pair[0]
            if ticker in self.known_tickers:
                found.add(ticker)
        for match in re.findall(r"\$([A-Z]{3,5})\b", text.upper()):
            if match in self.known_tickers:
                found.add(match)
        return list(found)

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            )
            self._model.to(self.device)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        return self._tokenizer

    def normalize_timestamp(self, timestamp: Any) -> int:
        """Normalise un timestamp en secondes depuis l'époque"""
        try:
            if isinstance(timestamp, int):
                if timestamp > 9999999999:
                    return int(timestamp / 1000)
                return timestamp
            if isinstance(timestamp, str):
                try:
                    ts = int(timestamp)
                    if ts > 9999999999:
                        return int(ts / 1000)
                    return ts
                except ValueError:
                    try:
                        dt = datetime.strptime(timestamp, "%a, %d %b %Y %H:%M:%S %z")
                        return int(dt.timestamp())
                    except ValueError:
                        try:
                            dt = datetime.strptime(timestamp, "%a, %d %b %Y %H:%M:%S")
                            return int(dt.timestamp())
                        except ValueError:
                            dt = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            return int(dt.timestamp())
            if isinstance(timestamp, datetime):
                return int(timestamp.timestamp())
            return int(datetime.now().timestamp())
        except Exception as e:
            self.logger.warning(f"Erreur normalisation timestamp {timestamp}: {str(e)}")
            return int(datetime.now().timestamp())

    def classify_risk_llm(self, title, text):
        """Classifie le risque d'une news via LLM"""
        try:
            result = self.llm_classifier(f"{title}. {text}", self.risk_labels)
            label = result["labels"][0] if result["labels"] else "Neutral"
            score = result["scores"][0] if result["scores"] else 0.0
            return label, score
        except Exception:
            return "Neutral", 0.0

    async def fetch_all_news(self) -> List[Dict]:
        async def fetch_source(
            session: aiohttp.ClientSession, source: Dict
        ) -> List[Dict]:
            try:
                async with session.get(
                    source["url"], timeout=self.conn_timeout
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        if source["type"] == "rss":
                            return self._parse_rss(content, source)
                        data = json.loads(content)
                        return self._parse_json(data, source)
            except Exception as e:
                self.logger.debug(f"[{source['name']}] Échec: {str(e)}")
            return []

        if self.news_buffer:
            try:
                connector = aiohttp.TCPConnector(ssl=self.ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    primary_sources = [
                        s for s in self.sources if s.get("priority", 1) == 1
                    ]
                    tasks = [
                        fetch_source(session, source) for source in primary_sources
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    valid_news = []
                    for source, result in zip(primary_sources, results):
                        if isinstance(result, list) and result:
                            valid_news.extend(result)
                            self.logger.info(
                                f"[{source['name']}] {len(result)} news récupérées"
                            )
                    if valid_news:
                        self.news_buffer = self.patch_news_list(valid_news)
                        return self.news_buffer
            except Exception as e:
                self.logger.warning(f"Échec mise à jour rapide: {str(e)}")
            self.logger.info("Utilisation du buffer existant")
            return self.news_buffer

        try:
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [fetch_source(session, source) for source in self.sources]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                valid_news = []
                for source, result in zip(self.sources, results):
                    if isinstance(result, list) and result:
                        valid_news.extend(result)
                        self.logger.info(
                            f"[{source['name']}] {len(result)} news récupérées"
                        )
                if valid_news:
                    self.news_buffer = self.patch_news_list(valid_news)
                    return self.news_buffer
        except Exception as e:
            self.logger.error(f"Échec récupération news: {str(e)}")
        return []

    def _parse_rss(self, content: str, source: Dict) -> List[Dict]:
        try:
            soup = BeautifulSoup(content, "xml")
            items = soup.find_all("item")
            return [self._parse_rss_item(item, source) for item in items]
        except Exception as e:
            self.logger.error(f"Error parsing RSS {source['name']}: {str(e)}")
            return []

    def analyze_sentiment_batch(
        self, news_items: List[Dict], low_watermark_ratio: float = None
    ) -> List[Dict]:
        news_items = self.patch_news_list(news_items)
        # PATCH ABSOLU : bloque et log toute valeur hors borne AVANT toute utilisation
        if low_watermark_ratio is None:
            low_watermark_ratio = getattr(self, "low_watermark_ratio", 0.75)
        try:
            low_watermark_ratio = float(low_watermark_ratio)
        except Exception:
            print(
                f"[DEBUG] Watermark ratio '{low_watermark_ratio}' non convertible, fallback à 0.75"
            )
            low_watermark_ratio = 0.75
        if low_watermark_ratio > 0.8 or low_watermark_ratio < 0.05:
            print(
                f"[DEBUG] Watermark ratio {low_watermark_ratio} hors borne, fallback à 0.75"
            )
            low_watermark_ratio = 0.75
        # Toujours borne stricte silencieuse
        low_watermark_ratio = min(max(low_watermark_ratio, 0.05), 0.8)
        if not news_items:
            print("[SENTIMENT] Aucun article à analyser.")
            return []
        try:
            texts = [f"{item['title']}. {item['text']}"[:512] for item in news_items]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            results = []
            for i, item in enumerate(news_items):
                sentiment = float(scores[i][2] - scores[i][0])
                label, risk_score = self.classify_risk_llm(
                    item.get("title", ""), item.get("text", "")
                )
                results.append(
                    {
                        **item,
                        "sentiment": sentiment,
                        "impact_score": 1.0,
                        "risk_class": label,
                        "risk_score": risk_score,
                    }
                )
            return results
        except Exception as e:
            print(
                f"[DEBUG] EXCEPTION analyze_sentiment_batch: {e} | watermark={low_watermark_ratio}"
            )
            self.logger.error(
                f"Error in sentiment analysis: {str(e)} | watermark={low_watermark_ratio}"
            )
            return []

    async def update_analysis(self):
        try:
            raw_news = await self.fetch_all_news()
            self.logger.debug(f"Fetched {len(raw_news)} raw news items")
            analyzed_news = self.analyze_sentiment_batch(raw_news)
            if not isinstance(analyzed_news, list):
                self.logger.error(
                    f"Invalid sentiment results type: {type(analyzed_news)}"
                )
                analyzed_news = []
            sentiment_scores = [
                n.get("sentiment", 0) for n in analyzed_news if isinstance(n, dict)
            ]
            mean_sentiment = (
                float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
            )
            std_sentiment = float(np.std(sentiment_scores)) if sentiment_scores else 0.0
            self.logger.info(
                f"News analysis: {len(analyzed_news)} items | Mean sentiment: {mean_sentiment:.4f} ± {std_sentiment:.4f}"
            )
            self.news_buffer = analyzed_news[-200:]
            summary = self.get_sentiment_summary()
            await self._save_state(
                {
                    "mean_sentiment": mean_sentiment,
                    "std_sentiment": std_sentiment,
                    "analyzed_news": analyzed_news[:50],
                    "sentiment_global": summary.get("sentiment_global", 0.0),
                    "top_symbols": summary.get("top_symbols", []),
                    "top_news": summary.get("top_news", []),
                }
            )
            return {
                "mean": mean_sentiment,
                "std": std_sentiment,
                "scores": sentiment_scores,
                "items": analyzed_news,
            }
        except Exception as e:
            self.logger.error(f"News update failed: {str(e)}", exc_info=True)
            return {"mean": 0.0, "std": 0.0, "scores": [], "items": []}

    def get_sentiment_summary(self, top_n=5):
        valid = [
            item
            for item in self.news_buffer
            if "sentiment" in item and item["sentiment"] is not None
        ]
        if not valid:
            return {
                "sentiment_global": 0.0,
                "n_news": 0,
                "top_symbols": [],
                "top_news": [],
            }
        sentiments = [item["sentiment"] for item in valid]
        sentiment_global = float(np.mean(sentiments))
        top_news = sorted(valid, key=lambda x: abs(x["sentiment"]), reverse=True)[
            :top_n
        ]
        top_news_titles = [news["title"] for news in top_news if "title" in news]
        symbol_scores = {}
        for item in valid:
            for s in item.get("symbols", []):
                symbol_scores.setdefault(s, []).append(item["sentiment"])
        top_symbols = sorted(
            symbol_scores.items(), key=lambda kv: abs(np.mean(kv[1])), reverse=True
        )
        top_symbols = [s for s, scores in top_symbols[:top_n]]
        return {
            "sentiment_global": sentiment_global,
            "n_news": len(valid),
            "top_symbols": top_symbols,
            "top_news": top_news_titles,
        }

    async def get_symbol_sentiment(
        self, symbol: str, news_list: Optional[list] = None
    ) -> float:
        try:
            symbol_key = symbol.replace("/", "").upper()
            coin_mapping = {
                "BTC": ["BTC", "BITCOIN", "XBT"],
                "ETH": ["ETH", "ETHEREUM", "ETHER"],
                "SOL": ["SOL", "SOLANA"],
                "ADA": ["ADA", "CARDANO"],
                "TRX": ["TRX", "TRON"],
                "BNB": ["BNB", "BINANCE", "BINANCECOIN"],
                "XRP": ["XRP", "RIPPLE"],
                "DOGE": ["DOGE", "DOGECOIN"],
                "AVAX": ["AVAX", "AVALANCHE"],
                "DOT": ["DOT", "POLKADOT"],
                "MATIC": ["MATIC", "POLYGON"],
                "LINK": ["LINK", "CHAINLINK"],
                "UNI": ["UNI", "UNISWAP"],
                "AAVE": ["AAVE"],
                "ATOM": ["ATOM", "COSMOS"],
                "NEAR": ["NEAR", "NEAR PROTOCOL"],
                "ALGO": ["ALGO", "ALGORAND"],
                "FTM": ["FTM", "FANTOM"],
                "XLM": ["XLM", "STELLAR"],
                "HBAR": ["HBAR", "HEDERA"],
            }
            coin = None
            for cm in sorted(coin_mapping.keys(), key=len, reverse=True):
                if symbol_key.startswith(cm):
                    coin = cm
                    break
            if coin is None:
                coin = symbol_key[:3]
            search_terms = coin_mapping.get(coin, [coin])
            if news_list is None:
                news_list = self.news_buffer
            total_sentiment = 0.0
            total_weight = 0.0
            matched_news = 0
            current_time = datetime.now(timezone.utc).timestamp()
            for news in news_list:
                try:
                    news_symbols = news.get("symbols", [])
                    title = news.get("title", "").lower()
                    text = news.get("text", "").lower()
                    content = f"{title} {text}"
                    timestamp = self.normalize_timestamp(
                        news.get("timestamp", current_time)
                    )
                    match_extracted = any(
                        s.upper().strip() in [term.upper() for term in search_terms]
                        for s in news_symbols
                    )
                    match_content = any(
                        term.lower() in content for term in search_terms
                    )
                    if match_extracted or match_content:
                        hours_old = (current_time - timestamp) / 3600
                        decay = 0.5 ** (hours_old / 24)
                        sentiment = float(news.get("sentiment", 0))
                        impact = float(news.get("impact_score", 1) or 1)
                        source_weight = float(news.get("source_weight", 0.7))
                        weight = decay * impact * source_weight
                        total_sentiment += sentiment * weight
                        total_weight += weight
                        matched_news += 1
                except Exception as e:
                    self.logger.warning(
                        f"Erreur traitement news pour {symbol}: {str(e)}"
                    )
                    continue
            if total_weight > 0:
                final_score = total_sentiment / total_weight
                self.logger.info(
                    f"Sentiment {symbol}: {final_score:.3f} (basé sur {matched_news} news)"
                )
                return float(max(min(final_score, 1.0), -1.0))
            else:
                self.logger.info(f"Pas de sentiment calculable pour {symbol}")
                return 0.0
        except Exception as e:
            self.logger.error(f"Erreur calcul sentiment pour {symbol}: {str(e)}")
            return 0.0

    def _extract_symbols_from_text(self, text):
        text = text.lower()
        found = set()
        for key, symbol in self.SYMBOL_MAPPING.items():
            if re.search(r"\b" + re.escape(key) + r"\b", text):
                found.add(symbol)
        return list(found)

    def patch_news_item(self, news):
        if "symbols" not in news or not news["symbols"]:
            title = news.get("title", "") or ""
            text = news.get("text", "") or ""
            symbols = self._extract_symbols_from_text(title + " " + text)
            news["symbols"] = symbols
        if "sentiment" not in news or news["sentiment"] is None:
            news["sentiment"] = 0.0
        else:
            try:
                news["sentiment"] = float(news["sentiment"])
            except Exception:
                news["sentiment"] = 0.0
        return news

    def patch_news_list(self, news_list):
        return [self.patch_news_item(news) for news in news_list]
