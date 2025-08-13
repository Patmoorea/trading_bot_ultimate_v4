"""
Sentiment Analyzer Module
"""
from typing import Dict, List, Union
from datetime import datetime
import logging
import random
class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sources = {
            'twitter': True,
            'reddit': True,
            'news': True
        }
        logging.info("Sentiment Analyzer initialized")
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment of a single text"""
        # Simplified sentiment analysis for testing
        sentiment = 'positive' if 'bullish' in text.lower() else 'negative'
        return {
            'sentiment': sentiment,
            'score': 0.85 if sentiment == 'positive' else 0.15,
            'polarity': 0.5,
            'subjectivity': 0.5
        }
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Analyze sentiment for a batch of texts"""
        return [self.analyze_text(text) for text in texts]
    def get_aggregated_sentiment(self, symbol: str) -> Dict[str, Union[str, float, Dict]]:
        """Get aggregated sentiment from all sources"""
        all_sentiments = []
        source_stats = {'twitter': 0, 'reddit': 0, 'news': 0}
        if self.sources['twitter']:
            twitter_sentiments = self._get_twitter_sentiment(symbol)
            all_sentiments.extend(twitter_sentiments)
            source_stats['twitter'] = len(twitter_sentiments)
        if self.sources['reddit']:
            reddit_sentiments = self._get_reddit_sentiment(symbol)
            all_sentiments.extend(reddit_sentiments)
            source_stats['reddit'] = len(reddit_sentiments)
        if self.sources['news']:
            news_sentiments = self._get_news_sentiment(symbol)
            all_sentiments.extend(news_sentiments)
            source_stats['news'] = len(news_sentiments)
        sample_size = len(all_sentiments)
        if sample_size == 0:
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'weighted_score': 0.0,
                'source_stats': source_stats,
                'sample_size': 0
            }
        positive_count = sum(1 for s in all_sentiments if s['sentiment'] == 'positive')
        confidence = positive_count / sample_size
        weighted_score = sum(s['score'] for s in all_sentiments) / sample_size
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'overall_sentiment': 'positive' if confidence > 0.5 else 'negative',
            'confidence': confidence,
            'weighted_score': weighted_score,
            'source_stats': source_stats,
            'sample_size': sample_size
        }
    def _get_twitter_sentiment(self, symbol: str) -> List[Dict]:
        """Get sentiment from Twitter"""
        texts = [
            f"Bullish on {symbol}!",
            f"{symbol} looking strong today",
            f"Not sure about {symbol}'s movement"
        ]
        return self.analyze_batch(texts)
    def _get_reddit_sentiment(self, symbol: str) -> List[Dict]:
        """Get sentiment from Reddit"""
        texts = [
            f"DD on {symbol}",
            f"What do you think about {symbol}?",
            f"{symbol} technical analysis"
        ]
        return self.analyze_batch(texts)
    def _get_news_sentiment(self, symbol: str) -> List[Dict]:
        """Get sentiment from news sources"""
        texts = [
            f"{symbol} reaches new high",
            f"Market analysis: {symbol}",
            f"Expert opinion on {symbol}"
        ]
        return self.analyze_batch(texts)
    def toggle_source(self, source: str, enabled: bool = True):
        """Enable or disable a sentiment source"""
        if source not in self.sources:
            raise ValueError(f"Invalid source: {source}")
        self.sources[source] = enabled
