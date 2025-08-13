from transformers import pipeline
class NewsAnalyzer:
    def __init__(self):
        try:
            self.analyzer = pipeline("sentiment-analysis")
        except Exception as e:
            print(f"Warning: Failed to initialize sentiment analyzer - {str(e)}")
            self.analyzer = None
    def analyze_news(self, text: str) -> float:
        if self.analyzer is None:
            return 0.5  # Valeur neutre par défaut
        result = self.analyzer(text)
        return result[0]['score'] if result[0]['label'] == 'POSITIVE' else 1 - result[0]['score']
