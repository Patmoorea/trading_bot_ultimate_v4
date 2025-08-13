from transformers import pipeline
class RealNewsAnalyzer:
    """Analyse de sentiment réel avec FinBERT"""
    def __init__(self):
        self.analyzer = pipeline(
            "text-classification",
            model="yiyanghkust/finbert-tone",
            tokenizer="yiyanghkust/finbert-tone"
        )
    def get_sentiment(self, text):
        return self.analyzer(text)[0]['label']  # POSITIVE/NEGATIVE/NEUTRAL
