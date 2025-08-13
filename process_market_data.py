    async def process_market_data(self):
        """
        Traitement et analyse des données de marché en temps réel
        Returns:
            tuple: (market_data, indicators_results) ou (None, None) en cas d'erreur
        """
        try:
            # 1. Récupération et validation des données
            market_data = self.buffer.get_latest()
            if not self._validate_market_data(market_data):
                logger.warning("❌ Données de marché invalides ou manquantes")
                await self.telegram.send_message("⚠️ Données de marché invalides - Vérification en cours...")
                return None, None
            # 2. Calcul parallèle des indicateurs
            indicators_results = await self._calculate_indicators_parallel(market_data)
            # 3. Analyse de la liquidité et création de la heatmap
            try:
                orderbook_data = await self.exchange.get_orderbook(config["TRADING"]["pairs"])
                heatmap = await self._generate_liquidity_heatmap(orderbook_data)
            except Exception as e:
                logger.error(f"Erreur génération heatmap: {e}")
                heatmap = None
            # 4. Détection et notification des signaux importants
            if important_signals := self._detect_important_signals(indicators_results):
                await self._send_signal_notifications(important_signals)
            # 5. Mise à jour du dashboard
            await self._update_dashboard(
                market_data=market_data,
                indicators=indicators_results,
                heatmap=heatmap,
            )
            return market_data, indicators_results
        except Exception as e:
            logger.error(f"❌ Erreur critique traitement données: {str(e)}")
            await self.telegram.send_message(
                f"🚨 Erreur critique:\n"
                f"Type: {type(e).__name__}\n"
                f"Details: {str(e)}\n"
            return None, None
    def _validate_market_data(self, data):
        """
        Valide l'intégrité et la qualité des données de marché
        """
        if not data:
            return False
        try:
            for timeframe in config["TRADING"]["timeframes"]:
                if timeframe not in data:
                    logger.error(f"Timeframe manquant: {timeframe}")
                    return False
                for pair in config["TRADING"]["pairs"]:
                    if pair not in data[timeframe]:
                        logger.error(f"Paire manquante: {pair} pour {timeframe}")
                        return False
                    df = data[timeframe][pair]
                    if df.empty or df.isnull().values.any():
                        logger.error(f"Données invalides pour {pair} - {timeframe}")
                        return False
            return True
        except Exception as e:
            logger.error(f"Erreur validation données: {e}")
            return False
    async def _calculate_indicators_parallel(self, market_data):
        """
        Calcule les indicateurs en parallèle pour optimiser les performances
        """
        indicators_results = {}
        async def process_timeframe(timeframe, data):
            try:
                indicators_results[timeframe] = self.advanced_indicators.analyze_timeframe(data, timeframe)
            except Exception as e:
                logger.error(f"Erreur calcul indicateurs {timeframe}: {e}")
                indicators_results[timeframe] = None
        # Création des tâches pour chaque timeframe
        tasks = [
            process_timeframe(timeframe, market_data[timeframe])
            for timeframe in config["TRADING"]["timeframes"]
        ]
        # Exécution en parallèle
        await asyncio.gather(*tasks)
        return indicators_results
    async def _generate_liquidity_heatmap(self, orderbook_data):
        """
        Génère une heatmap de liquidité optimisée
        """
        try:
            return await asyncio.create_task(
                self.generate_heatmap(orderbook_data)
            )
        except Exception as e:
            logger.error(f"Erreur génération heatmap: {e}")
            return None
    def _detect_important_signals(self, indicators_results):
        """
        Détecte les signaux importants nécessitant une notification
        """
        important_signals = []
        for timeframe, indicators in indicators_results.items():
            if not indicators:
                continue
            # Analyse des signaux de tendance
            if trend_signal := self._analyze_trend_signals(indicators["trend"]):
                important_signals.append({
                    "timeframe": timeframe,
                    "type": "trend",
                    "signal": trend_signal
                })
            # Analyse des signaux de volatilité
            if vol_signal := self._analyze_volatility_signals(indicators["volatility"]):
                important_signals.append({
                    "timeframe": timeframe,
                    "type": "volatility",
                    "signal": vol_signal
                })
            # Analyse des signaux de volume
            if vol_signal := self._analyze_volume_signals(indicators["volume"]):
                important_signals.append({
                    "timeframe": timeframe,
                    "type": "volume",
                    "signal": vol_signal
                })
        return important_signals
    async def _send_signal_notifications(self, signals):
        """
        Envoie des notifications pour les signaux importants
        """
        for signal in signals:
            message = (
                f"🔔 Signal Important Détecté!\n"
                f"⏰ Timeframe: {signal['timeframe']}\n"
                f"📊 Type: {signal['type']}\n"
                f"💡 Signal: {signal['signal']}\n"
            await self.telegram.send_message(message)
