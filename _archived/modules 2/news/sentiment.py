# ===== INTEGRATION FINBERT =====
from transformers import pipeline
def analyze_with_finbert(text):
    """Extension du module existant"""
    analyzer = pipeline("text-classification", model="yiyanghkust/finbert-tone")
    return analyzer(text)[0]
from transformers import pipeline
def analyze_news(text):
    """Analyse de sentiment avec FinBERT"""
    analyzer = pipeline("text-classification", 
                      model="yiyanghkust/finbert-tone")
    return analyzer(text)[0]['label']
def get_news_impact_score(headline):
    """Extension de l'analyse existante"""
    sentiment = analyze_news(headline)
    return 1.0 if sentiment == 'POSITIVE' else -1.0 if sentiment == 'NEGATIVE' else 0.0
class AdvancedSentimentAnalyzer:
    """Complément d'analyse avec FinBERT sans écraser l'existant"""
    def __init__(self):
        try:
            from transformers import BertForSequenceClassification, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
            self.available = True
        except ImportError:
            self.available = False
    def analyze(self, text):
        if not self.available:
            return None
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        return {
            'finbert_scores': outputs.logits.softmax(dim=1).tolist()[0],
            'version': 'finbert-v1'
        }
# ===== NOUVEAUX COMPOSANTS ===== #
def check_finbert_availability():
    """Vérifie si FinBERT peut être chargé"""
    try:
        from transformers import BertForSequenceClassification
        return True
    except:
        return False
if check_finbert_availability():
    FinBERT = AdvancedSentimentAnalyzer()
else:
    logging.warning("FinBERT non disponible - utilisation du analyseur de base")
def get_enhanced_sentiment(text):
    """Version optimisée utilisant le core"""
    from src.core_merged.news.sentiment import EnhancedNewsAnalyzer
    analyzer = EnhancedNewsAnalyzer()
    try:
        result = analyzer.analyze(text)
        return {
            'sentiment': 'POSITIVE' if result[0][1] > 0.7 else 'NEGATIVE' if result[0][1] < 0.3 else 'NEUTRAL',
            'confidence': float(result[0][1])
        }
    except Exception as e:
        logger.warning(f"Fallback to basic analysis: {str(e)}")
        return analyze_with_finbert(text)  # Conserve l'ancienne méthode
