"""
Module unifié de traitement des actualités - Version 2.2
"""
import sys
from pathlib import Path
# Configuration du chemin
module_dir = Path(__file__).parent
project_root = module_dir.parent.parent
sys.path.append(str(project_root))
# Import des composants de base
try:
    from sentiment_news import SentimentAnalyzer as BaseSentimentAnalyzer
except ImportError:
    print("Warning: Base SentimentAnalyzer not found")
    class BaseSentimentAnalyzer: pass
try:
    from sentiment_influence import NewsImpactEvaluator as BaseImpactEvaluator
except ImportError:
    print("Warning: Base NewsImpactEvaluator not found")
    class BaseImpactEvaluator: pass
# Import des composants avancés
try:
    from .sentiment import SentimentProcessor
except ImportError:
    print("Warning: SentimentProcessor not found")
    class SentimentProcessor: pass
try:
    from .real_news_processor import RealNewsProcessor
except ImportError:
    print("Warning: RealNewsProcessor not found")
    class RealNewsProcessor: pass
class EnhancedNewsProcessor(
    BaseSentimentAnalyzer,
    BaseImpactEvaluator,
    SentimentProcessor,
    RealNewsProcessor
):
    """Classe unifiée avec fallback pour composants manquants"""
    def __init__(self):
        # Initialisation conditionnelle des parents
        if hasattr(BaseSentimentAnalyzer, '__init__'):
            BaseSentimentAnalyzer.__init__(self)
        if hasattr(BaseImpactEvaluator, '__init__'):
            BaseImpactEvaluator.__init__(self)
        self.initialized = True
    def check_components(self):
        """Vérifie quels composants sont actifs"""
        return {
            'BaseSentiment': isinstance(self, BaseSentimentAnalyzer),
            'ImpactEvaluator': isinstance(self, BaseImpactEvaluator),
            'SentimentProcessor': isinstance(self, SentimentProcessor),
            'RealNewsProcessor': isinstance(self, RealNewsProcessor)
        }
