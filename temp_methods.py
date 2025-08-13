import datetime
import logging

logger = logging.getLogger(__name__)


async def process_market_data(self):
    """Traitement des données de marché avec tous les indicateurs et préparation du rapport Telegram"""
    try:
        # 1. Récupération des données de marché
        market_data = self.buffer.get_latest()
        if market_data is None or not market_data:
            logger.warning("Données de marché manquantes")
            return None, None, None

        # 2. Calcul de tous les indicateurs pour chaque timeframe
        indicators_results = {}
        for timeframe in self.config["TRADING"]["timeframes"]:
            try:
                tf_data = market_data[timeframe]
                indicators_results[timeframe] = (
                    self.advanced_indicators.analyze_timeframe(tf_data, timeframe)
                )
            except Exception as e:
                logger.error(f"Erreur calcul indicateurs {timeframe}: {e}")
                indicators_results[timeframe] = {
                    "trend": {"trend_strength": 0},
                    "volatility": {"current_volatility": 0},
                    "volume": {"volume_profile": {"strength": "N/A"}},
                    "dominant_signal": "Neutre",
                }

        # 3. Génération de la heatmap de liquidité (optionnelle)
        try:
            orderbook = await self.exchange.get_orderbook(
                self.config["TRADING"]["pairs"]
            )
            heatmap = self.generate_heatmap(orderbook)
        except Exception as e:
            logger.error(f"Erreur génération heatmap: {e}")
            heatmap = None

        # 4. Notification des signaux importants
        await self._notify_significant_signals(indicators_results)

        # 5. Mise à jour du dashboard en temps réel
        self.dashboard.update(
            market_data,
            indicators_results,
            heatmap,
        )

        # 6. (Nouveau) Calcul des décisions de trade pour chaque timeframe
        trade_decisions = {}
        for timeframe in self.config["TRADING"]["timeframes"]:
            try:
                # À adapter selon ta logique de décision :
                decision = await self.generate_trade_decision(
                    market_data.get(timeframe, {}),
                    indicators_results.get(timeframe, {}),
                    timeframe,
                )
                if decision:
                    # Ajoute tous les champs souhaités
                    trade_decisions[timeframe] = {
                        "action": decision.get("action", "NEUTRAL"),
                        "confidence": decision.get("confidence", 0),
                        "tech": decision.get("tech", 0),
                        "ai": decision.get("ai", 0),
                        "sentiment": decision.get("sentiment", 0),
                    }
            except Exception as e:
                logger.warning(f"Erreur génération décision {timeframe}: {e}")

        # 7. Récupération du sentiment/news (optionnel)
        try:
            news_sentiment = await self.news_analyzer.analyze_recent_news()
        except Exception as e:
            logger.warning(f"Erreur récupération news: {e}")
            news_sentiment = None

        # 8. Récupération du régime courant (optionnel)
        try:
            regime = self.regime_detector.detect_regime(indicators_results)
        except Exception as e:
            logger.warning(f"Erreur détection régime: {e}")
            regime = "Indéterminé"

        # 9. Préparation du rapport Telegram
        report = self._generate_analysis_report(
            indicators_results,
            regime,
            news_sentiment=news_sentiment,
            trade_decisions=trade_decisions,
        )

        # 10. Envoi du rapport sur Telegram
        await self.telegram.send_message(report)

        return market_data, indicators_results, trade_decisions

    except Exception as e:
        logger.error(f"Erreur lors du traitement des données: {e}")
        await self.telegram.send_message(f"⚠️ Erreur traitement: {str(e)}")
        return None, None, None


def _generate_analysis_report(
    self, indicators_analysis, regime, news_sentiment=None, trade_decisions=None
):
    """Génère un rapport d'analyse détaillé avec news et décisions de trade"""
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    report = [
        "📊 Analyse complète du marché:",
        f"Date: {current_time} UTC",
        f"Régime: {regime}",
        "\nTendances principales:",
    ]
    # Ajout de l'analyse des news si disponible
    if news_sentiment:
        try:
            report.extend(
                [
                    "\n📰 Analyse des News:",
                    f"Sentiment: {news_sentiment.get('overall_sentiment', 0):.2%}",
                    f"Impact estimé: {news_sentiment.get('impact_score', 0):.2%}",
                    f"Événements majeurs: {news_sentiment.get('major_events', 'Aucun')}",
                ]
            )
        except Exception as e:
            logger.warning(f"Erreur traitement news: {e}")
    else:
        report.append("\n📰 Analyse des News: Aucune donnée disponible.")

    # Ajout des dernières news si disponible
    major_news = news_sentiment.get("latest_news", []) if news_sentiment else []
    if major_news:
        report.append("Dernières news :")
        for news in major_news[:3]:
            report.append(f"- {news}")

    # Analyse par timeframe
    for timeframe, analysis in indicators_analysis.items():
        try:
            report.append(f"\n⏰ {timeframe}:")
            trend_strength = analysis.get("trend", {}).get("trend_strength", 0)
            volatility = analysis.get("volatility", {}).get("current_volatility", 0)
            volume_profile = analysis.get("volume", {}).get("volume_profile", {})
            report.extend(
                [
                    f"- Force de la tendance: {trend_strength:.2%}",
                    f"- Volatilité: {volatility:.2%}",
                    f"- Volume: {volume_profile.get('strength', 'N/A')}",
                    f"- Signal dominant: {analysis.get('dominant_signal', 'Neutre')}",
                ]
            )
            # Ajout de la décision de trade si disponible
            if trade_decisions and timeframe in trade_decisions:
                dec = trade_decisions[timeframe]
                report.append(
                    f"└─ 🎯 Décision de trade: {dec['action'].upper()} "
                    f"(Conf: {dec['confidence']:.2f}, "
                    f"Tech: {dec.get('tech',0):.2f}, "
                    f"IA: {dec.get('ai',0):.2f}, "
                    f"Sentiment: {dec.get('sentiment',0):.2f})"
                )
        except Exception as e:
            logger.warning(f"Erreur analyse timeframe {timeframe}: {e}")
            report.extend(
                [
                    f"\n⏰ {timeframe}:",
                    "- Données non disponibles",
                    "- Analyse en cours...",
                ]
            )
    return "\n".join(report)
