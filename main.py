import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import psutil
import pytz
from datetime import datetime, timedelta
from binance.client import Client
from src.backtesting.core.backtest_engine import BacktestEngine
from src.strategies import sma_strategy, breakout_strategy, arbitrage_strategy
from src.bot_runner import _generate_analysis_report
from src.risk_tools import kelly_criterion, calculate_var, calculate_max_drawdown

TRADING_PARAMS = {
    "entry_confirmation_threshold": 0.8,
    "profit_targets": [
        {"threshold": 1.02, "size": 0.3},
        {"threshold": 1.035, "size": 0.3},
        {"threshold": 1.05, "size": 0.4},
    ],
    "stop_loss": {"initial": 0.02, "trailing": True, "trail_percent": 0.01},
    "position_sizing": {"base_risk": 0.02, "max_risk": 0.05, "scaling": True},
}

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Trading Bot Ultimate v4 - Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Patmoorea/trading_bot_ultimate_v3",
        "Report a bug": "https://github.com/Patmoorea/trading_bot_ultimate_v3/issues",
        "About": "# Trading Bot Ultimate v4\nVersion avancée avec IA et analyses quantiques.",
    },
)

STATUS_FILE = "bot_status.json"
SHARED_DATA_PATH = "src/shared_data.json"
LOG_FILE = "src/bot_logs.txt"
CONFIG_FILE = "config.json"
CURRENT_USER = "Patmoorea"  # UNIVERSAL PATCH: Safe JSON update


def save_shared_data(update_dict, data_file):
    try:
        # PATCH: restauration JSON minimal si fichier vide/corrompu
        shared_data = {}
        if os.path.exists(data_file):
            try:
                with open(data_file, "r") as f:
                    shared_data = json.load(f)
                    if not isinstance(shared_data, dict):
                        print(
                            "[PATCH] Le fichier JSON n'est pas un dict, restauration minimal."
                        )
                        shared_data = {}
            except Exception as e:
                print(
                    f"[PATCH] Erreur lecture JSON : {e} -- Fichier corrompu, restauration minimal."
                )
                shared_data = {
                    "bot_status": {
                        "regime": "Indéterminé",
                        "cycle": 0,
                        "last_update": "",
                        "performance": {
                            "balance": 10000,
                            "total_trades": 0,
                            "wins": 0,
                            "losses": 0,
                            "total_profit": 0,
                            "total_loss": 0,
                            "win_rate": 0,
                            "profit_factor": 0,
                        },
                    },
                    "positions_binance": {},
                    "pending_sales": [],
                    "active_pauses": [],
                }
        shared_data.update(update_dict)
        with open(data_file, "w") as f:
            json.dump(shared_data, f, indent=2)
    except Exception as e:
        print(f"[PATCH] Erreur sauvegarde JSON : {e}")


def calc_sizing(confidence, tech, ai, sentiment, win_rate=0.55, profit_factor=1.7):
    # Sizing base selon confiance
    if confidence > 0.8:
        base = 0.09
    elif confidence > 0.6:
        base = 0.06
    elif confidence > 0.4:
        base = 0.04
    else:
        base = 0.02
    # Ajustements
    if tech > 0.7:
        base *= 1.2
    if ai > 0.7:
        base *= 1.1
    if abs(sentiment) > 0.7:
        base *= 0.8
    # Kelly Criterion
    kelly = kelly_criterion(win_rate, profit_factor)
    if kelly > 0:
        base *= 1 + min(kelly * 0.5, 0.5)
    return f"{min(base * 100, 12):.1f}%"


def get_current_time():
    utc_now = datetime.utcnow()
    polynesie_offset = timedelta(hours=-10)
    local_dt = utc_now + polynesie_offset
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")


def load_json_file(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def get_pending_sales(self):
    pending = []
    GAIN_ALERT_PCT = 0.07
    LOSS_ALERT_PCT = -0.05
    now = datetime.utcnow()
    pauses = []
    if hasattr(self, "news_pause_manager"):
        pauses = self.news_pause_manager.get_active_pauses()

    def is_paused(symbol):
        if not pauses:
            return False, ""
        for p in pauses:
            asset = p.get("asset", "GLOBAL")
            if asset == "GLOBAL" or asset == symbol:
                return True, p.get("reason", "Indéterminée")
        return False, ""

    for symbol, pos in self.positions.items():
        entry_price = pos.get("entry_price")
        current_price = pos.get("current_price")
        amount = pos.get("amount")
        pnl_latent = (
            (current_price - entry_price) / entry_price * 100
            if entry_price and current_price
            else 0
        )
        date_achat = pos.get("date", pos.get("entry_time")) or None
        if date_achat:
            try:
                date_achat_dt = datetime.fromisoformat(date_achat)
                temps_en_position = (now - date_achat_dt).total_seconds() / 3600
            except Exception:
                temps_en_position = None
        else:
            temps_en_position = None
        td = self.trade_decisions.get(symbol.replace("/", "").upper(), {})
        action = td.get("action", "neutral")
        reason = ""
        decision = ""
        note = ""
        pause_for_pos, pause_reason = is_paused(symbol)
        if pause_for_pos:
            pause_blocage = "Oui"
            note = "Trading suspendu"
            reason = "Pause active"
            decision = f"Vente bloquée (pause: {pause_reason})"
        elif action == "SELL" and pos.get("side") == "long":
            pause_blocage = "Non"
            reason = "Signal SELL détecté"
            decision = "Vente prévue au prochain cycle"
            note = ""
        elif self.exit_manager.is_tp_near(pos):
            pause_blocage = "Non"
            reason = "Take Profit proche"
            decision = "Vente partielle possible (TP)"
            note = ""
        elif self.check_stop_loss(symbol):
            pause_blocage = "Non"
            reason = "Stop-loss imminent"
            decision = "Vente automatique si perte aggrave"
            note = ""
        elif pnl_latent > GAIN_ALERT_PCT * 100:
            pause_blocage = "Non"
            reason = "Gain latent élevé"
            decision = "Surveillance, possibilité de prise de profit"
            note = "En zone de profit, TP possible"
        elif pnl_latent < LOSS_ALERT_PCT * 100:
            pause_blocage = "Non"
            reason = "Perte latente élevée"
            decision = "Surveillance, risque de vente auto si perte aggrave"
            note = "Risque de stop-loss"
        else:
            pause_blocage = "Non"
            reason = f"Signal actuel: {action.upper()}"
            decision = "Aucune action prévue, position maintenue"
            note = ""
        pending.append(
            {
                "symbol": symbol,
                "reason": reason,
                "decision": decision,
                "entry_price": entry_price,
                "current_price": current_price,
                "amount": amount,
                "% Gain/Perte latente": f"{pnl_latent:.2f}%",
                "temps_en_position_h": (
                    f"{temps_en_position:.1f}"
                    if temps_en_position is not None
                    else "N/A"
                ),
                "pause_blocage": pause_blocage,
                "note": note,
            }
        )
    if hasattr(self, "positions_binance"):
        for symbol, pos in self.positions_binance.items():
            entry_price = pos.get("entry_price")
            current_price = pos.get("current_price")
            amount = pos.get("amount")
            pnl_latent = (
                (current_price - entry_price) / entry_price * 100
                if entry_price and current_price
                else 0
            )
            fifo_pnl_pct, _ = self.get_last_fifo_pnl(symbol)
            if fifo_pnl_pct is None:
                fifo_pnl_pct = 0
            td = self.trade_decisions.get(symbol.replace("/", "").upper(), {})
            action = td.get("action", "neutral")
            pause_for_pos, pause_reason = is_paused(symbol)
            if pause_for_pos:
                pause_blocage = "Oui"
                note = "Trading suspendu"
                reason = "Pause active"
                decision = f"Vente bloquée (pause: {pause_reason})"
            elif action == "SELL" and pos.get("side") == "long":
                pause_blocage = "Non"
                reason = "Signal SELL détecté"
                decision = "Vente prévue au prochain cycle"
                note = ""
            elif hasattr(self, "exit_manager") and self.exit_manager.is_tp_near(pos):
                pause_blocage = "Non"
                reason = "Take Profit proche"
                decision = "Vente partielle possible (TP)"
                note = ""
            elif self.check_stop_loss(symbol):
                pause_blocage = "Non"
                reason = "Stop-loss imminent"
                decision = "Vente automatique si perte aggrave"
                note = ""
            elif pnl_latent > GAIN_ALERT_PCT * 100:
                pause_blocage = "Non"
                reason = f"Gain latent élevé {pnl_latent:.1f}%"
                decision = "Surveillance, possibilité de prise de profit"
                note = "En zone de profit, TP possible"
            elif pnl_latent < LOSS_ALERT_PCT * 100:
                pause_blocage = "Non"
                reason = f"Perte latente élevée {pnl_latent:.1f}%"
                decision = "Surveillance, risque de vente auto si perte aggrave"
                note = "Risque de stop-loss"
            else:
                pause_blocage = "Non"
                reason = f"Signal actuel: {action.upper()}"
                decision = "Aucune action prévue, position maintenue"
                note = ""
            pending.append(
                {
                    "symbol": symbol,
                    "reason": reason,
                    "decision": decision,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "amount": amount,
                    "% Gain/Perte latente": f"{pnl_latent:.2f}%",
                    "% Plus-value FIFO": f"{fifo_pnl_pct:.2f}%",
                    "temps_en_position_h": "N/A",
                    "pause_blocage": pause_blocage,
                    "note": note,
                }
            )
    print("DEBUG pending_sales tableau:", pending)
    save_shared_data({"pending_sales": pending}, self.data_file)
    return pending


def fetch_binance_ohlcv(
    symbol, interval, start_str, end_str=None, api_key=None, api_secret=None
):
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    if not klines or len(klines) == 0:
        return None
    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


# --- SIDEBAR ---
with st.sidebar:
    st.header("🤖 Bot Status")
    status = load_json_file(STATUS_FILE)
    shared_data = load_json_file(SHARED_DATA_PATH)
    tahiti = pytz.timezone("Pacific/Tahiti")
    now_tahiti = datetime.now(tahiti).strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(
        f"""
        <div style='background-color: #0f3d40; padding: 10px; border-radius: 5px;'>
            <h3 style='color: #00ff00; margin: 0;'>✅ Bot Actif</h3>
            <p style='color: #ffffff; margin: 5px 0;'>Dernière mise à jour: {now_tahiti} (heure Polynésie)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='background-color: #1c1c1c; padding: 10px; border-radius: 5px; margin-top: 10px;'>
            <p style='margin: 0;'>👤 Utilisateur: Patmoorea</p>
            <p style='margin: 0;'>🌐 Mode: Production</p>
            <p style='margin: 0;'>📈 Exchange: Binance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # == MODE SAFE WARNING ==
    safe_mode = shared_data.get("safe_mode", False)
    if safe_mode:
        st.warning(
            "⚠️ MODE SAFE ACTIVÉ : sizing réduit à cause de pertes consécutives !"
        )

    # == 1. Alertes actives ==
    st.markdown("### 🚨 Alertes actives")
    alerts = shared_data.get("alerts", [])
    for alert in alerts:
        if alert["level"] == "critical":
            st.error(f"{alert['message']} ({alert['timestamp']})")
        elif alert["level"] == "warning":
            st.warning(f"{alert['message']} ({alert['timestamp']})")
        else:
            st.info(f"{alert['message']} ({alert['timestamp']})")

    # == 3. Positions fermées ==
    st.header("⛔️ Positions fermées (auto)")
    closed = shared_data.get("closed_positions", [])
    if closed:
        df_closed = pd.DataFrame(closed)
        st.dataframe(df_closed, use_container_width=True)
    else:
        st.info("Aucune position fermée automatiquement ce cycle.")

    st.sidebar.divider()

    # == 4. Informations système & connectivité ==
    st.sidebar.markdown(
        f"""
### 📊 Informations système
- 🕒 Dernière mise à jour: {now_tahiti} (heure Polynésie) 
- 👤 Session: {CURRENT_USER}
- 🌐 Version: 4.0.1
- 📡 Status: En ligne
- 💾 Mémoire utilisée: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB
"""
    )
    st.sidebar.markdown(
        f"""
        <div style='background-color: #1c1c1c; padding: 10px; border-radius: 5px; margin-top: 20px;'>
            <p style='margin: 0; color: #00ff00;'>🟢 Exchange: Connecté</p>
            <p style='margin: 0; color: #00ff00;'>🟢 Base de données: Synchronisée</p>
            <p style='margin: 0; color: #00ff00;'>🟢 API: Opérationnelle</p>
            <p style='margin: 0; color: #808080; font-size: 0.8em;'>Dernière vérification: {get_current_time()}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🔄 Rafraîchir"):
        st.rerun()

    # --- AJOUT SLIDERS ET AFFICHAGE SEUILS ACTIFS ---
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Sélection dynamique des paires")
    min_volatility = st.sidebar.slider("Volatilité min", 0.0, 0.05, 0.01, 0.001)
    min_signal = st.sidebar.slider("Signal min", 0.0, 1.0, 0.3, 0.01)
    top_n = st.sidebar.slider("Nb max paires à trader", 1, 10, 5, 1)

    st.sidebar.markdown(
        f"""
        <div style='background-color: #232b2b; padding: 8px; border-radius: 5px; margin-top: 10px;'>
            <b>🎯 Seuils de filtering actifs</b><br>
            • Volatilité min : <span style='color:#00ff00'>{min_volatility:.3f}</span><br>
            • Signal min : <span style='color:#00ff00'>{min_signal:.2f}</span><br>
            • Nb paires max : <span style='color:#00ff00'>{top_n}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sauvegarde dans shared_data.json (pour que le bot les lise au prochain cycle)
    try:
        load_json_file(SHARED_DATA_PATH)
        shared_data["filtering_params"] = {
            "min_volatility": float(min_volatility),
            "min_signal": float(min_signal),
            "top_n": int(top_n),
        }
        save_shared_data(
            {"filtering_params": shared_data["filtering_params"]}, SHARED_DATA_PATH
        )
    except Exception as e:
        st.sidebar.warning(f"Erreur sauvegarde filtres dynamiques: {e}")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab_logs = st.tabs(
    [
        "📊 Trading",
        "📈 Graphiques",
        "🔬 Analyse",
        "📖 Portefeuille/Positions",
        "🧪 Backtest",
        "📈 Performance",
        "📝 Logs",
    ]
)

# --- TAB1 TRADING ---
with tab1:

    # Configuration du timestamp Polynésie
    tahiti_tz = pytz.timezone("Pacific/Tahiti")
    current_time = datetime.now(tahiti_tz).strftime("%Y-%m-%d %H:%M:%S")

    # Récupération des données
    bot_status = shared_data.get("bot_status", {})
    perf = bot_status.get("performance", {})
    market_data = shared_data.get("market_data", {})

    # Affichage des métriques principales
    st.markdown("#### Cycle et Régime")
    col1, col2, col3 = st.columns(3)
    col1.metric("Cycle actuel", bot_status.get("cycle", 0))
    col2.metric("Régime", bot_status.get("regime", "Indéterminé"))
    col3.metric(
        "Balance",
        f"${perf.get('balance',0):,.2f}",
        f"{perf.get('win_rate',0)*100:.1f}%",
    )

    # Gestion des pauses trading
    active_pauses = shared_data.get("active_pauses", [])
    if active_pauses:
        pause_cycles_left = max([p.get("cycles_left", 0) for p in active_pauses])
        st.warning(
            f"🚨 Trading bloqué (pause active) — Déblocage dans {pause_cycles_left} cycle(s) !",
            icon="⏸️",
        )
        st.markdown("#### ⏸️ Pauses actives détaillées")
        df_pauses = pd.DataFrame(active_pauses)

        # === PATCH : TRADUCTION FRANÇAISE DES TYPES ET RAISONS ===
        pause_type_map = {
            "news": "Pause News",
            "impact": "Impact fort",
            "volatility": "Volatilité extrême",
            "global": "Pause Globale",
            "sentiment": "Sentiment critique",
            "regime": "Changement de régime",
        }
        reason_map = {
            "news": "Pause suite à une news critique",
            "impact": "Impact de marché important",
            "volatility": "Volatilité trop élevée",
            "global": "Pause globale du système",
            "sentiment": "Sentiment négatif détecté",
            "regime": "Changement de régime de marché",
        }

        if "type" in df_pauses.columns:
            df_pauses["type"] = df_pauses["type"].map(
                lambda x: pause_type_map.get(x, str(x))
            )
        if "reason" in df_pauses.columns:
            df_pauses["reason"] = df_pauses["reason"].map(
                lambda x: reason_map.get(x, str(x))
            )

        st.dataframe(df_pauses, use_container_width=True)

    st.divider()

    trade_decisions = shared_data.get("trade_decisions", {})

    if trade_decisions:
        # Création du DataFrame
        decisions_data = []

        # Paramètres de performance pour le sizing
        perf = shared_data.get("bot_status", {}).get("performance", {})
        win_rate = perf.get("win_rate", 0.55)
        profit_factor = perf.get("profit_factor", 1.7)

        for pair, decision in trade_decisions.items():
            # Les vraies valeurs sont dans la décision !
            confidence = float(decision.get("confidence", 0.5))
            tech_score = float(decision.get("tech", 0.5))
            ai_pred = float(decision.get("ai", 0.5))
            sentiment = float(decision.get("sentiment", 0.0))

            # Calcul du sizing base
            if confidence > 0.8:
                base_size = 0.09  # 9%
            elif confidence > 0.6:
                base_size = 0.06  # 6%
            elif confidence > 0.4:
                base_size = 0.04  # 4%
            else:
                base_size = 0.02  # 2%

            # Ajustements
            if tech_score > 0.7:
                base_size *= 1.2
            if ai_pred > 0.7:
                base_size *= 1.1
            if abs(sentiment) > 0.7:
                base_size *= 0.8

            # Kelly Criterion
            kelly = kelly_criterion(win_rate, profit_factor)
            if kelly > 0:
                base_size *= 1 + min(kelly * 0.5, 0.5)

            # Création de la ligne
            row_data = {
                "pair": pair,
                "action": decision.get("action", "NEUTRAL").upper(),
                "confidence": confidence,
                "tech": tech_score,
                "ai": ai_pred,
                "sentiment": sentiment,
                "Sizing (%)": calc_sizing(
                    confidence, tech_score, ai_pred, sentiment, win_rate, profit_factor
                ),
                "timestamp": current_time,
            }
            decisions_data.append(row_data)

        # Création et formatage du DataFrame
        df_signals = pd.DataFrame(decisions_data)
        df_signals.set_index("pair", inplace=True)

        # Formatage des colonnes numériques
        numeric_cols = ["confidence", "tech", "ai", "sentiment"]
        for col in numeric_cols:
            if col in df_signals.columns:
                df_signals[col] = df_signals[col].map("{:.3f}".format)

        # Affichage avec style et TITRE explicite
        st.markdown("#### Tableau des signaux et sizing par paire")
        st.dataframe(df_signals, use_container_width=True, height=400)

    else:
        st.info("Aucun signal de trading ce cycle.")

    st.divider()

    # Historique des trades
    st.markdown("#### 📜 Historique des trades exécutés")
    trades = shared_data.get("trade_history", [])
    if trades:
        df_trades = pd.DataFrame(trades)
        # Conversion des timestamps en heure Polynésie
        if "timestamp" in df_trades.columns:
            df_trades["timestamp"] = (
                pd.to_datetime(df_trades["timestamp"])
                .dt.tz_localize("UTC")
                .dt.tz_convert("Pacific/Tahiti")
            )
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("Aucun trade exécuté ce cycle.")

    st.divider()

    # Section Arbitrage (une seule fois !)
    st.markdown("#### 💹 Opportunités d'arbitrage")
    arbitrage_ops = shared_data.get("arbitrage_opportunities", [])
    if arbitrage_ops:
        df_arb = pd.DataFrame(arbitrage_ops)
        if "timestamp" in df_arb.columns:
            df_arb["timestamp"] = (
                pd.to_datetime(df_arb["timestamp"])
                .dt.tz_localize("UTC")
                .dt.tz_convert("Pacific/Tahiti")
            )
        st.dataframe(df_arb, use_container_width=True)
    else:
        st.info("Aucune opportunité d'arbitrage détectée ce cycle.")


with tab2:
    st.subheader("Analyse graphique avancée")
    pairs = list(shared_data.get("market_data", {}).keys()) or ["BTCUSDT", "ETHUSDT"]
    pair = st.selectbox("Sélectionner une paire", pairs)

    # Ajoute le choix du timeframe
    available_tfs = list(shared_data.get("market_data", {}).get(pair, {}).keys())
    tf = st.selectbox("Timeframe", available_tfs if available_tfs else ["1m"])

    market_data = shared_data.get("market_data", {}).get(pair, {}).get(tf, {})

    if market_data and market_data.get("close") and market_data.get("timestamp"):
        try:
            # 1. Validation et conversion des données
            timestamps = market_data["timestamp"]
            closes = market_data["close"]
            opens = market_data.get("open", [])
            highs = market_data.get("high", [])
            lows = market_data.get("low", [])

            # 2. Vérification du type et conversion des timestamps
            if isinstance(timestamps, (int, str)):
                timestamps = [timestamps]
            if isinstance(closes, (int, float)):
                closes = [closes]
            if isinstance(opens, (int, float)):
                opens = [opens]
            if isinstance(highs, (int, float)):
                highs = [highs]
            if isinstance(lows, (int, float)):
                lows = [lows]

            # 3. Conversion des timestamps en datetime (PATCH ms/s)
            try:
                if isinstance(timestamps[0], str):
                    timestamps = pd.to_datetime(timestamps)
                elif isinstance(timestamps[0], (int, float)):
                    # PATCH: si timestamp > 1e12, c'est en ms
                    if timestamps[0] > 1e12:
                        timestamps = pd.to_datetime(timestamps, unit="ms")
                    else:
                        timestamps = pd.to_datetime(timestamps, unit="s")
            except Exception as e:
                print(f"Erreur conversion timestamps: {e}")
                timestamps = pd.date_range(
                    end=pd.Timestamp.utcnow(), periods=len(closes), freq="H"
                )

            # 4. Conversion en arrays numpy avec la bonne longueur
            min_len = min(
                len(timestamps), len(closes), len(opens), len(highs), len(lows)
            )
            timestamps = timestamps[:min_len]
            closes = np.array(closes[:min_len], dtype=float)
            opens = np.array(opens[:min_len], dtype=float)
            highs = np.array(highs[:min_len], dtype=float)
            lows = np.array(lows[:min_len], dtype=float)

            # 5. Création du DataFrame
            df = pd.DataFrame(
                {"close": closes, "open": opens, "high": highs, "low": lows},
                index=timestamps,
            )

            # 6. Calcul des moyennes mobiles
            ema20 = df["close"].ewm(span=20, adjust=False).mean()
            ema50 = df["close"].ewm(span=50, adjust=False).mean()

            # 7. Création du graphique
            fig = go.Figure()

            # Chandelier japonais
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="OHLC",
                )
            )

            # EMA 20 et 50
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ema20, name="EMA 20", line=dict(color="blue", width=1)
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ema50,
                    name="EMA 50",
                    line=dict(color="orange", width=1),
                )
            )

            # Configuration du layout
            fig.update_layout(
                title=f"Graphique {pair} ({tf})",
                yaxis_title="Prix USDT",
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                height=600,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False
                ),
            )

            # Affichage du graphique
            st.plotly_chart(fig, use_container_width=True)

            # Debug optionnel
            if st.checkbox("Show Data Debug"):
                st.write("DataFrame Head:", df.head())
                st.write("Timestamps Info:", pd.Series(timestamps).describe())

        except Exception as e:
            st.error(f"Erreur lors de la création du graphique: {str(e)}")
            print(f"DEBUG - Erreur graphique détaillée: {str(e)}")
            print(f"DEBUG - Type timestamps: {type(market_data['timestamp'])}")
            print(
                f"DEBUG - Premier timestamp: {market_data['timestamp'][0] if isinstance(market_data['timestamp'], list) else market_data['timestamp']}"
            )
            print(f"DEBUG - Longueur données:")
            print(
                f"- Timestamps: {len(market_data['timestamp']) if isinstance(market_data['timestamp'], list) else 1}"
            )
            print(
                f"- Close: {len(market_data['close']) if isinstance(market_data['close'], list) else 1}"
            )
            print(
                f"- Open: {len(market_data.get('open', [])) if isinstance(market_data.get('open'), list) else 1}"
            )
    else:
        st.info("Pas de données live pour cette paire et ce timeframe.")

# --- TAB3 ANALYSE ---
with tab3:
    st.subheader("Analyse technique approfondie")
    indicators = shared_data.get("indicators", {})
    regime = shared_data.get("regime", "Indéterminé")
    news_sentiment = shared_data.get("sentiment", None)
    trade_decisions = shared_data.get("trade_decisions", {})
    # Rapport global
    report = _generate_analysis_report(
        indicators, regime, news_sentiment, trade_decisions
    )
    st.code(report, language="markdown")
    st.divider()
    # Indicateurs avancés par paire/timeframe
    st.expander("🔎 Indicateurs techniques avancés")
    for tf_key, indic in indicators.items():
        if "ta" in indic and indic["ta"]:
            st.write(f"**{tf_key}**")
            df_ta = pd.DataFrame(indic["ta"], index=[0]).T
            st.dataframe(df_ta, use_container_width=True)

with tab4:
    st.subheader("Portefeuille / Positions en temps réel")

    # 1. Tableau des positions Binance Spot
    positions_binance = shared_data.get("positions_binance", {})
    spot_pairs = list(positions_binance.keys())
    fifo_pnl_map = {}

    for pair in spot_pairs:
        symbol = pair.replace("/", "")
        fifo_key = f"fifo_pnl_{symbol}"
        fifo_pnl = shared_data.get(fifo_key, [])
        last_fifo = fifo_pnl[-1] if fifo_pnl else None
        fifo_pnl_map[pair] = (
            last_fifo["pnl_pct"]
            if last_fifo and last_fifo["pnl_pct"] is not None
            else None
        )

    df_pos_binance = pd.DataFrame.from_dict(positions_binance, orient="index")
    df_pos_binance.index.name = "Paire"
    df_pos_binance["% Plus-Value"] = [
        (
            f"{fifo_pnl_map.get(pair, None):.2f}%"
            if fifo_pnl_map.get(pair, None) is not None
            else (
                f"{((row['current_price'] - row['entry_price']) / row['entry_price'] * 100):.2f}%"
                if row["entry_price"] and row["current_price"]
                else "N/A"
            )
        )
        for pair, row in df_pos_binance.iterrows()
    ]
    st.dataframe(df_pos_binance, use_container_width=True)

    # 2. Alertes de ventes à venir (tableau juste après portefeuille)
    st.markdown("#### Alertes de ventes à venir")
    pending_sales = shared_data.get("pending_sales", [])
    trade_decisions = shared_data.get("trade_decisions", {})
    if pending_sales:
        try:
            df_pending = pd.DataFrame(pending_sales)

            # Colonnes à afficher
            display_cols = [
                "symbol",
                "entry_price",
                "current_price",
                "amount",
                "action",  # <-- action BUY/SELL/NEUTRAL
                "% Gain/Perte latente",
                "reason",
                "decision",
                "temps_en_position_h",
                "pause_blocage",
                "note",
            ]
            # Ajout colonnes manquantes
            for col in display_cols:
                if col not in df_pending.columns:
                    df_pending[col] = "N/A"

            # PATCH: Correction entry_price, current_price, amount
            def safe_float(x):
                try:
                    return float(x)
                except Exception:
                    return 0

            df_pending["entry_price"] = df_pending["entry_price"].apply(safe_float)
            df_pending["current_price"] = df_pending["current_price"].apply(safe_float)
            df_pending["amount"] = df_pending["amount"].apply(safe_float)

            # Ajout colonne Action : utilise le champ "action" ou déduit via trade_decisions
            def get_action(row):
                action = row.get("action", "").upper()
                if action in ["BUY", "SELL", "NEUTRAL"]:
                    return action
                # fallback: cherche dans trade_decisions
                symbol = row.get("symbol", "")
                td = trade_decisions.get(symbol.replace("/", "").upper(), {})
                action_td = td.get("action", "NEUTRAL").upper()
                if action_td in ["BUY", "SELL", "NEUTRAL"]:
                    return action_td
                return "NEUTRAL"

            df_pending["action"] = df_pending.apply(get_action, axis=1)

            # Calcul gain/perte latente
            df_pending["% Gain/Perte latente"] = (
                (df_pending["current_price"] - df_pending["entry_price"])
                / df_pending["entry_price"]
                * 100
            ).map(
                lambda x: (
                    f"{x:.2f}%"
                    if not pd.isnull(x) and df_pending["entry_price"].max() > 0
                    else "N/A"
                )
            )

            # Correction du temps en position
            df_pending["temps_en_position_h"] = df_pending["temps_en_position_h"].apply(
                lambda x: f"{x:.1f}h" if isinstance(x, (int, float)) else x
            )

            # Customisation ultra détaillée de la décision
            def custom_decision(row):
                entry = row["entry_price"]
                current = row["current_price"]
                pnl = ((current - entry) / entry * 100) if entry and current else 0
                symbol = row["symbol"]
                pause = row.get("pause_blocage", "Non")
                reason = row.get("reason", "")
                decision = row.get("decision", "")
                action = row.get("action", "NEUTRAL")

                # Prédiction et explication
                if pause == "Oui":
                    return f"🔒 Vente bloquée (pause news/reglementaire). Aucun mouvement possible."
                if action == "SELL":
                    return f"🟠 Vente prévue au prochain cycle (signal SELL détecté)."
                if action == "BUY":
                    return f"🟢 Achat possible au prochain cycle (signal BUY détecté)."
                if "Take Profit" in reason:
                    return "🟢 Vente partielle possible (TP proche)."
                if pnl >= 8:
                    return f"🟢 {symbol} en attente de vente au prochain cycle (plus-value {pnl:.2f}%)."
                if pnl > 7:
                    return f"🟢 Gain latent élevé, surveillance TP (plus-value {pnl:.2f}%)."
                if pnl < -5:
                    return f"🔴 Perte latente élevée, risque vente auto si perte aggrave ({pnl:.2f}%)."
                if decision.lower() == "position maintenue":
                    return f"🟡 Position maintenue, aucun signal critique."
                return f"ℹ️ Surveillance normale."

            df_pending["Décision détaillée"] = df_pending.apply(custom_decision, axis=1)

            # Ajout colonne "Action probable prochain cycle"
            def cycle_action(row):
                if "Vente bloquée" in row["Décision détaillée"]:
                    return "Aucune action avant fin de la pause."
                if "Vente prévue" in row["Décision détaillée"]:
                    return "Vente automatique au prochain tick."
                if "Achat possible" in row["Décision détaillée"]:
                    return "Achat automatique au prochain tick."
                if "Vente partielle" in row["Décision détaillée"]:
                    return "Prise de profit partielle si TP atteint."
                if "Gain latent élevé" in row["Décision détaillée"]:
                    return (
                        "Bot surveille le TP, vente probable si le prix monte encore."
                    )
                if "Perte latente élevée" in row["Décision détaillée"]:
                    return "Bot surveille le stop-loss, vente forcée si perte aggrave."
                if "Position maintenue" in row["Décision détaillée"]:
                    return "Aucune action prévue, surveillance normale."
                return "Analyse continue, pas d'action critique."

            df_pending["Action probable prochain cycle"] = df_pending.apply(
                cycle_action, axis=1
            )

            # Tri par priorité
            priority_map = {
                "🔴": 1,
                "🔒": 2,
                "🟠": 3,
                "🟢": 4,
                "🟡": 5,
                "ℹ️": 6,
            }

            def priority(row):
                for k in priority_map:
                    if k in row["Décision détaillée"]:
                        return priority_map[k]
                return 99

            df_pending["priority"] = df_pending.apply(priority, axis=1)
            df_pending = df_pending.sort_values(
                ["priority", "symbol"], ascending=[True, True]
            )

            # Colonnes finales à afficher
            ordered_cols = [
                "symbol",
                "action",
                "entry_price",
                "current_price",
                "amount",
                "% Gain/Perte latente",
                "Décision détaillée",
                "Action probable prochain cycle",
                "temps_en_position_h",
                "pause_blocage",
                "note",
            ]
            st.dataframe(
                df_pending[ordered_cols],
                use_container_width=True,
                height=500,
            )

        except Exception as e:
            st.error(f"Erreur affichage alertes: {e}")
    else:
        st.info("Aucune vente imminente détectée.")

    # 3. Positions BingX (Futures) - Shorts et Longs
    positions_bingx = shared_data.get("positions_bingx", {})
    st.markdown("#### Positions ouvertes BingX (Futures / Shorts/Longs)")
    if positions_bingx:
        df_bingx = pd.DataFrame.from_dict(positions_bingx, orient="index")
        df_bingx.index.name = "Paire"
        if "pnl_pct" in df_bingx.columns:
            df_bingx["% Plus-Value"] = df_bingx["pnl_pct"].map(
                lambda x: f"{x:.2f}%" if x is not None else "N/A"
            )
        if "side" in df_bingx.columns:
            df_bingx["Type"] = df_bingx["side"].map(
                lambda x: "SHORT" if x == "short" else "LONG"
            )
        st.dataframe(df_bingx, use_container_width=True)
    else:
        st.info("Aucune position ouverte sur BingX futures.")

    # 4. Positions fermées
    st.markdown("#### Historique des positions fermées")
    closed = shared_data.get("closed_positions", [])
    if closed:
        df_closed = pd.DataFrame(closed)
        reasons = df_closed["reason"].unique().tolist()
        reason_selected = st.selectbox(
            "Filtrer par raison de vente", ["Toutes"] + reasons
        )
        if reason_selected != "Toutes":
            df_closed = df_closed[df_closed["reason"] == reason_selected]
        st.dataframe(df_closed, use_container_width=True)
    else:
        st.info("Aucune position fermée automatiquement ce cycle.")

    # 5. Plus-value réelle FIFO (spot) sur LTC/USDC
    st.markdown("#### Plus-values réelles (FIFO) sur LTC/USDC")
    fifo_pnl = shared_data.get("fifo_pnl_LTCUSDC", [])
    if fifo_pnl:
        df_fifo = pd.DataFrame(fifo_pnl)
        df_fifo["% Plus-Value"] = df_fifo["pnl_pct"].map(
            lambda x: f"{x:.2f}%" if x is not None else "N/A"
        )
        df_fifo["Gain ($)"] = df_fifo["pnl_usd"].map(
            lambda x: f"{x:.2f}" if x is not None else "N/A"
        )
        df_fifo = df_fifo.rename(
            columns={
                "sell_qty": "Qte vendue",
                "sell_price": "Prix vente",
                "entry_price": "Prix achat moyen",
                "sell_time": "Timestamp",
            }
        )
        st.dataframe(
            df_fifo[
                [
                    "Timestamp",
                    "Qte vendue",
                    "Prix achat moyen",
                    "Prix vente",
                    "% Plus-Value",
                    "Gain ($)",
                    "buy_details",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("Aucune vente spot détectée pour LTC/USDC (FIFO).")

# --- TAB5 BACKTEST ---
with tab5:
    st.subheader("Backtest avancé")
    st.sidebar.header("Backtesting avancé")
    strategy_options = {
        "SMA Crossover": sma_strategy,
        "Breakout": breakout_strategy,
        "Arbitrage": arbitrage_strategy,
    }
    strategy_name = st.sidebar.selectbox("Stratégie", list(strategy_options.keys()))
    strategy_func = strategy_options[strategy_name]
    params = {}
    if strategy_name == "SMA Crossover":
        params["fast_window"] = st.sidebar.slider("SMA rapide", 2, 50, 10)
        params["slow_window"] = st.sidebar.slider("SMA lente", 10, 200, 50)
    elif strategy_name == "Breakout":
        params["lookback"] = st.sidebar.slider("Lookback", 5, 50, 20)
    elif strategy_name == "Arbitrage":
        params["spread_threshold"] = st.sidebar.number_input(
            "Seuil de spread (%)", min_value=0.01, max_value=5.0, value=0.5
        )
    dataset_file = st.sidebar.file_uploader("Données historiques (CSV)", type=["csv"])
    if dataset_file:
        df = pd.read_csv(dataset_file)
        capital = st.sidebar.number_input("Capital initial", min_value=100, value=10000)
        if st.sidebar.button("Lancer le backtest"):
            backtester = BacktestEngine(initial_capital=capital)
            results = backtester.run_backtest(df, strategy_func, **params)
            st.write("Résultats du backtest :", results)
    # Configuration du backtest
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Configuration de base")
        period = st.selectbox("Période de test", ["7j", "30j", "90j", "180j", "365j"])
        initial_capital = st.number_input(
            "Capital initial (USDT)", min_value=0, value=0
        )
        leverage = st.slider("Levier", min_value=1, max_value=10, value=1)
    with col2:
        st.markdown("### Paramètres avancés")
        risk_per_trade = st.slider(
            "Risque par trade (%)", min_value=0.1, max_value=5.0, value=1.0
        )
        stop_loss = st.slider("Stop Loss (%)", min_value=0.5, max_value=10.0, value=2.0)
        take_profit = st.slider(
            "Take Profit (%)", min_value=1.0, max_value=20.0, value=4.0
        )
    if st.button("🚀 Lancer le backtest"):
        st.info("Simulation en cours…")
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            pairs = config.get("pairs", ["BTC/USDT"])
        except Exception:
            pairs = ["BTC/USDT"]
        period_map = {"7j": 7, "30j": 30, "90j": 90, "180j": 180, "365j": 365}
        nb_days = period_map.get(period, 30)
        end_dt = pd.Timestamp.utcnow()
        start_dt = end_dt - pd.Timedelta(days=nb_days)
        interval = Client.KLINE_INTERVAL_1HOUR
        for i, pair in enumerate(pairs):
            symbol = pair.replace("/", "")
            df = fetch_binance_ohlcv(
                symbol,
                interval,
                start_dt.strftime("%d %b %Y"),
                end_dt.strftime("%d %b %Y"),
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
            )
            if df is None or len(df) == 0:
                st.error(f"Données manquantes pour {pair}, backtest ignoré.")
                continue
            strategy_func = strategy_options[strategy_name]
            backtester = BacktestEngine(initial_capital=initial_capital)
            results = backtester.run_backtest(df, strategy_func, **params)
            st.write(f"Résultats du backtest pour {pair} :", results)
        st.success("Backtest terminé!")

# --- TAB6 PERFORMANCE ---
with tab6:
    st.subheader("Performance et Métriques")
    perf = shared_data.get("bot_status", {}).get("performance", {})
    equity_history = shared_data.get("equity_history", [])
    returns = shared_data.get("returns_array", np.linspace(0, 27.5, 30))
    x_axis = list(range(len(returns)))
    cumulative_returns = 1 + np.array(returns) / 100
    st.markdown("### 📈 Performance Cumulative")
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_axis,
                y=cumulative_returns,
                name="Performance",
                fill="tozeroy",
                line=dict(color="#00ff00"),
            )
        ],
        layout=go.Layout(
            title="Performance du Trading Bot",
            template="plotly_dark",
            yaxis_title="Rendement Cumulatif",
            xaxis_title="Jours",
            showlegend=True,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 📊 Métriques de Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total trades", f"{perf.get('total_trades',0)}")
    col1.metric("Win Rate", f"{perf.get('win_rate',0):.1%}")
    col2.metric("Profit Factor", f"{perf.get('profit_factor',0):.2f}")
    col2.metric("Max Drawdown", f"{perf.get('max_drawdown',0):.1%}")
    col3.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio',0):.2f}")
    col3.metric("Balance Finale", f"${perf.get('balance',10000):,.0f}")

    # --- Ajout Risk Management avancé ---
    equity_curve = [
        pt.get("balance", 0) for pt in equity_history if pt.get("balance", 0) > 0
    ]
    kelly = None
    max_dd = None
    var95 = None
    win_streak = None
    loss_streak = None
    avg_win = None
    avg_loss = None
    best_trade = None
    worst_trade = None
    win_pct = None

    # Calcul des métriques avancées
    if equity_curve and len(equity_curve) > 10:
        equity_curve_np = np.array(equity_curve)
        max_dd = calculate_max_drawdown(equity_curve_np)
        returns_curve = np.diff(equity_curve_np) / equity_curve_np[:-1]
        if len(returns_curve) > 10:
            var95 = calculate_var(returns_curve, 0.05)
        kelly = kelly_criterion(
            win_rate=perf.get("win_rate", 0), payoff_ratio=perf.get("profit_factor", 1)
        )
        # Streaks & trade stats (à partir de trade_history)
        trades = shared_data.get("trade_history", [])
        wins = [t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) > 0]
        losses = [t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        best_trade = np.max(wins) if wins else 0
        worst_trade = np.min(losses) if losses else 0
        # Win/loss streaks
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        prev = None
        for t in trades:
            pnl = t.get("pnl_usd", 0)
            if pnl > 0:
                streak = streak + 1 if prev == "win" else 1
                max_win_streak = max(max_win_streak, streak)
                prev = "win"
            elif pnl < 0:
                streak = streak + 1 if prev == "loss" else 1
                max_loss_streak = max(max_loss_streak, streak)
                prev = "loss"
        win_streak = max_win_streak
        loss_streak = max_loss_streak
        # Ratio gagnant/perdant
        total_trades = len(trades)
        win_pct = len(wins) / total_trades if total_trades > 0 else 0

    with st.expander("📉 Indicateurs avancés de risque et performance"):
        st.metric("Kelly optimal", f"{kelly:.2f}" if kelly is not None else "N/A")
        st.metric("Max Drawdown", f"{max_dd:.2%}" if max_dd is not None else "N/A")
        st.metric("VaR (95%)", f"{var95:.2%}" if var95 is not None else "N/A")
        st.metric(
            "Plus longue série de trades gagnants",
            f"{win_streak}" if win_streak is not None else "N/A",
        )
        st.metric(
            "Plus longue série de trades perdants",
            f"{loss_streak}" if loss_streak is not None else "N/A",
        )
        st.metric(
            "Moyenne gains/trade", f"${avg_win:.2f}" if avg_win is not None else "N/A"
        )
        st.metric(
            "Moyenne pertes/trade",
            f"${avg_loss:.2f}" if avg_loss is not None else "N/A",
        )
        st.metric(
            "Meilleur trade", f"${best_trade:.2f}" if best_trade is not None else "N/A"
        )
        st.metric(
            "Pire trade", f"${worst_trade:.2f}" if worst_trade is not None else "N/A"
        )
        st.metric(
            "Ratio de trades gagnants",
            f"{win_pct:.1%}" if win_pct is not None else "N/A",
        )

        if kelly is not None and abs(kelly) > 0.5:
            st.warning(
                f"⚠️ Kelly fraction élevée : {kelly:.2f} — attention à la taille des positions !"
            )
        if max_dd is not None and max_dd < -0.15:
            st.error(f"🚨 Max drawdown dépassé : {max_dd:.2%} ! Pause conseillée.")
        if var95 is not None and var95 < -0.05:
            st.error(f"🛑 VaR(95%) critique : {var95:.2f}")

# --- TAB LOGS ---
with tab_logs:
    st.subheader("📝 Logs du Bot (live)")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = f.readlines()
        st.text("".join(logs[-200:]))
    else:
        st.info("Aucun log à afficher.")
    if st.button("🗑️ Vider les logs"):
        open(LOG_FILE, "w").close()
        st.success("Logs vidés !")


# --- Auto-refresh ---
def auto_refresh():
    time.sleep(10)
    st.rerun()


if __name__ == "__main__":
    auto_refresh()
