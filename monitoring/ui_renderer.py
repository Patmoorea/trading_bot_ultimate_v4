class UIRenderer:
    """Classe dédiée au rendu de l'interface utilisateur"""
    @staticmethod
    async def render_portfolio_tab(bot):
        """Rendu de l'onglet Portfolio"""
        if st.session_state.bot_running:
            try:
                portfolio = st.session_state.get("portfolio")
                if portfolio:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "💰 Total Value",
                            f"{portfolio.get('total_value', 0):.2f} USDC",
                            f"{portfolio.get('daily_pnl', 0):+.2f} USDC",
                        )
                    with col2:
                        st.metric(
                            "📈 24h Volume",
                            f"{portfolio.get('volume_24h', 0):.2f} USDC",
                            f"{portfolio.get('volume_change', 0):+.2f}%",
                        )
                    with col3:
                        positions = portfolio.get("positions", [])
                        st.metric(
                            "🔄 Active Positions",
                            str(len(positions)),
                            f"{len(positions)} active",
                        )
                    if positions:
                        st.subheader("Active Positions")
                        st.dataframe(pd.DataFrame(positions), use_container_width=True)
                    else:
                        st.info("💡 No active positions")
                else:
                    st.warning("⚠️ Waiting for portfolio data...")
            except Exception as e:
                st.error(f"❌ Portfolio error: {str(e)}")
        else:
            st.warning("⚠️ Start trading to view portfolio")
    @staticmethod
    async def render_trading_tab(bot):
        """Rendu de l'onglet Trading"""
        if st.session_state.bot_running:
            try:
                latest_data = bot.latest_data.get("BTCUSDT", {})
                if latest_data:
                    UIRenderer._render_market_metrics(latest_data)
                if bot.indicators:
                    st.subheader("Trading Signals")
                    st.dataframe(pd.DataFrame(bot.indicators), use_container_width=True)
                else:
                    st.info("💡 Waiting for signals...")
            except Exception as e:
                st.error(f"❌ Trading data error: {str(e)}")
        else:
            st.warning("⚠️ Start trading to view signals")
    @staticmethod
    async def render_analysis_tab(bot):
        """Rendu de l'onglet Analysis"""
        if st.session_state.bot_running:
            try:
                UIRenderer._render_technical_analysis(bot)
                UIRenderer._render_quantum_signals(bot)
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
        else:
            st.warning("⚠️ Start trading to view analysis")
    @staticmethod
    def _render_market_metrics(latest_data):
        """Rendu des métriques de marché"""
        col1, col2 = st.columns(2)
        with col1:
            current_price = latest_data[-1]["close"]
            prev_price = latest_data[-2]["close"] if len(latest_data) > 1 else current_price
            price_change = ((current_price - prev_price) / prev_price * 100) if prev_price else 0
            st.metric("BTC/USDC Price", f"{current_price:.2f}", f"{price_change:+.2f}%")
        with col2:
            current_vol = latest_data[-1]["volume"]
            prev_vol = latest_data[-2]["volume"] if len(latest_data) > 1 else current_vol
            vol_change = ((current_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
            st.metric("Trading Volume", f"{current_vol:.2f}", f"{vol_change:+.2f}%")
    @staticmethod
    def _render_technical_analysis(bot):
        """Rendu de l'analyse technique"""
        if bot.latest_data and bot.indicators:
            st.subheader("Technical Analysis")
            for symbol in bot.latest_data:
                process_market_data(bot, symbol)
            if hasattr(bot, "advanced_indicators"):
                analysis = bot.advanced_indicators.get_all_signals()
                st.dataframe(pd.DataFrame(analysis), use_container_width=True)
        else:
            st.info("💡 Waiting for market data...")
    @staticmethod
    def _render_quantum_signals(bot):
        """Rendu des signaux quantiques"""
        if hasattr(bot, "qsvm") and bot.qsvm is not None:
            try:
                features = bot.latest_data
                quantum_signal = bot.qsvm.predict(features)
                st.subheader("Quantum SVM Signal")
                st.metric("Quantum SVM Signal", quantum_signal)
            except Exception as e:
                st.warning(f"Erreur Quantum SVM : {e}")
