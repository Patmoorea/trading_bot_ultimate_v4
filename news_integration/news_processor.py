import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import asyncio
import aiohttp
from datetime import datetime, timedelta
class NewsProcessor:
    def __init__(self):
        # Chargement du modèle FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.model.eval()
        # Sources d'actualités
        self.news_sources = [
            'https://api.newsapi.org',
            'https://api.cryptopanic.com',
            'https://api.coindesk.com',
            'https://api.cointelegraph.com',
            'https://api.theblockcrypto.com',
            'https://api.decrypt.co',
            'https://api.bitcoinmagazine.com',
            'https://api.coinjournal.net',
            'https://api.cryptoslate.com',
            'https://api.cryptobriefing.com',
            'https://api.bitcoinist.com',
            'https://api.newsbtc.com'
        ]
    async def process_news(self) -> Dict:
        """Process principal des news"""
        # Récupération des news
        news = await self.fetch_news()
        # Analyse du sentiment
        analyzed_news = self.analyze_sentiment(news)
        # Agrégation des impacts
        impact = self._aggregate_impact(analyzed_news)
        return {
            'news': analyzed_news,
            'overall_impact': impact,
            'timestamp': datetime.utcnow()
        }
    async def fetch_news(self) -> List[Dict]:
        """Récupère les actualités de toutes les sources en parallèle"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_source(session, source) for source in self.news_sources]
            results = await asyncio.gather(*tasks)
        # Fusion et déduplication
        all_news = []
        seen_urls = set()
        for source_news in results:
            for news in source_news:
                if news['url'] not in seen_urls:
                    all_news.append(news)
                    seen_urls.add(news['url'])
        return all_news
    def analyze_sentiment(self, news: List[Dict]) -> List[Dict]:
        """Analyse le sentiment avec FinBERT"""
        results = []
        for article in news:
            # Tokenization
            text = f"{article['title']} {article['description']}"
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            # Prédiction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            sentiment = {
                'positive': float(scores[0][0]),
                'neutral': float(scores[0][1]),
                'negative': float(scores[0][2])
            }
            # Calcul impact ~0.2s
            impact = self._calculate_impact(article, sentiment)
            results.append({
                **article,
                'sentiment': sentiment,
                'impact': impact
            })
        return results
    def _calculate_impact(self, article: Dict, sentiment: Dict) -> float:
        """Calcul de l'impact potentiel sur le marché"""
        # Poids des facteurs
        source_weight = self._get_source_weight(article['source'])
        time_weight = self._get_time_weight(article['published_at'])
        reach_weight = self._get_reach_weight(article)
        # Score de sentiment (-1 à 1)
        if sentiment['positive'] > 0.5:
            sent_score = sentiment['positive']
        elif sentiment['negative'] > 0.5:
            sent_score = -sentiment['negative']
        else:
            sent_score = 0
        # Impact final
        impact = sent_score * source_weight * time_weight * reach_weight
        return impact
    def _aggregate_impact(self, analyzed_news: List[Dict]) -> float:
        """Agrège l'impact de toutes les news"""
        if not analyzed_news:
            return 0.0
        impacts = [news['impact'] for news in analyzed_news]
        # Moyenne pondérée par la fraîcheur
        weights = [
            self._get_time_weight(news['published_at'])
            for news in analyzed_news
        ]
        return np.average(impacts, weights=weights)
    async def _fetch_source(self, session: aiohttp.ClientSession, source: str) -> List[Dict]:
        """Récupère les news d'une source"""
        try:
            async with session.get(source) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_news(data, source)
                return []
        except Exception as e:
            print(f"Erreur fetch {source}: {e}")
            return []
    def _parse_news(self, data: Dict, source: str) -> List[Dict]:
        """Parse les news selon le format de la source"""
        news = []
        # Parser spécifique selon la source
        if 'newsapi.org' in source:
            for article in data.get('articles', []):
                news.append({
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'source': article['source']['name'],
                    'published_at': datetime.fromisoformat(article['publishedAt']),
                    'shares': 0,
                    'comments': 0
                })
        elif 'cryptopanic.com' in source:
            for post in data.get('results', []):
                news.append({
                    'title': post['title'],
                    'description': post.get('description', ''),
                    'url': post['url'],
                    'source': post['source']['domain'],
                    'published_at': datetime.fromisoformat(post['published_at']),
                    'votes': post.get('votes', {}).get('positive', 0),
                    'comments': len(post.get('comments', []))
                })
        # Autres sources...
        return news
    def _get_source_weight(self, source: str) -> float:
        """Poids selon fiabilité de la source"""
        weights = {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'coindesk': 0.8,
            'cointelegraph': 0.7,
            'theblockcrypto': 0.8,
            'decrypt': 0.7
        }
        return weights.get(source.lower(), 0.5)
    def _get_time_weight(self, published_at: datetime) -> float:
        """Poids selon fraîcheur de la news"""
        if age < timedelta(hours=1):
            return 1.0
        elif age < timedelta(hours=6):
            return 0.8
        elif age < timedelta(hours=24):
            return 0.5
        else:
            return 0.2
    def _get_reach_weight(self, article: Dict) -> float:
        """Poids selon portée de l'article"""
        reach = 0.5
        if 'shares' in article:
            reach += min(article['shares'] / 1000, 0.3)
        if 'comments' in article:
            reach += min(article['comments'] / 100, 0.2)
        return min(reach, 1.0)
