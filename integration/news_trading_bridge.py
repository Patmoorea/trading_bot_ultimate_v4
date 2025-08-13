class NewsTradingBridge:
    def __init__(self):
        self.news_processor = OptimizedNewsProcessor()
        self.trading_engine = TradingEngine()
        self.strategy_adapter = StrategyAdapter()
    async def process_news_impact(self, news_event):
        """Intégration améliorée news-trading"""
        # Analyse rapide de l'impact
        impact = await self.news_processor.process_news(news_event)
        if impact['significance'] > 0.7:
            # Adaptation stratégie en temps réel
            strategy_updates = self.strategy_adapter.adapt_to_news(
                impact,
                current_market_state=self.get_market_state()
            )
            # Mise à jour trading en temps réel
            await self.trading_engine.update_strategy(strategy_updates)
            # Log et monitoring
            self.monitor.log_strategy_update(
                source='news',
                impact=impact,
                updates=strategy_updates
            )
