import re
import os
import json
import shutil
from datetime import datetime
from transformers import pipeline


class NewsPauseManager:
    # Criticité : mot-clé associé à une durée de pause par défaut (en cycles)
    CRITICAL_KEYWORDS = {
        "hack": 30,
        "exploit": 30,
        "theft": 20,
        "attack": 15,
        "scam": 15,
        "rug": 15,
        "exit scam": 15,
        "compromised": 12,
        "security breach": 12,
        "arrest": 10,
        "frozen": 10,
        "liquidation": 10,
        "insolvency": 10,
        "lawsuit": 8,
        "investigation": 8,
        "ban": 15,
        "delist": 8,
        "paused": 6,
        "halted": 6,
        "regulation": 10,
        "suspension": 6,
    }

    def __init__(self, default_pause_cycles=5, alert_callback=None, config=None):
        self.default_pause_cycles = default_pause_cycles
        self.config = config or {}
        self.sentiment_thresholds = self.config.get("news", {}).get(
            "sentiment_thresholds",
            {"neg": -0.7, "pos": 0.7, "impact": 0.6, "sources": 2, "risk_score": 0.7},
        )
        self.global_cycles_remaining = 0  # Pause globale
        self.last_event_time = None
        self.last_event_news = None
        self.last_triggered_title = None
        self.alert_callback = alert_callback
        self.pair_pauses = {}  # {pair: cycles_restants}
        self.buy_paused_pairs = set()  # Paires où seuls les achats sont bloqués
        self.active_pauses = []
        self.volatility_thresholds = {"low": 0.02, "medium": 0.05, "high": 0.08}
        self.market_conditions = {}

    def scan_news(self, news_list):
        print(f"\n[NEWSPAUSE DEBUG] Analyse de {len(news_list)} news")
        if not news_list:
            print("[NEWSPAUSE DEBUG] Liste de news vide")
            return False
        triggered = False
        for i, news in enumerate(news_list):
            print(f"\n[NEWSPAUSE DEBUG] Analyse news #{i+1}:")
            print(f"- Title: {news.get('title', 'NO TITLE')}")
            print(f"- Text: {news.get('text', 'NO TEXT')[:100]}...")
            print(f"- Sentiment: {news.get('sentiment', 'NO SENTIMENT')}")
            print(f"- Symbols: {news.get('symbols', []) or news.get('assets', [])}")
            print(f"- Processed: {news.get('processed', False)}")
            if news.get("processed"):
                print("➡️ News déjà traitée, skip")
                continue
            title = news.get("title", "") or ""
            text = news.get("text", "") or ""
            content = f"{title} {text}".lower()
            symbols = news.get("symbols", []) or news.get("assets", [])
            sentiment = float(news.get("sentiment", 0)) if "sentiment" in news else None
            impact = float(news.get("impact_score", 0)) if "impact_score" in news else 0
            n_sources = int(news.get("n_sources", 1))
            risk_class = news.get("risk_class", "Neutral")
            risk_score = float(news.get("risk_score", 0))
            thresholds = self.sentiment_thresholds

            # --- IA: Classification avancée ---
            if (
                risk_class
                in ["Hack", "Security", "Scam", "Regulatory", "Crash", "Pump"]
                and risk_score > thresholds["risk_score"]
            ):
                print(
                    f"⚠️ IA: Catégorie '{risk_class}' à risque + score élevé -> PAUSE GLOBALE"
                )
                self.global_cycles_remaining = 12
                news["processed"] = True
                self.last_event_time = datetime.now()
                self.last_event_news = news
                self.last_triggered_title = title
                triggered = True
                continue

            # --- Sentiment extrême + impact ---
            if sentiment is not None:
                if sentiment < thresholds["neg"] and impact > thresholds["impact"]:
                    print(
                        "⚠️ IA: Sentiment très négatif + impact élevé -> PAUSE GLOBALE"
                    )
                    self.global_cycles_remaining = 10
                    news["processed"] = True
                    self.last_event_time = datetime.now()
                    self.last_event_news = news
                    self.last_triggered_title = title
                    triggered = True
                    continue
                if sentiment > thresholds["pos"]:
                    print("🟢 IA: Sentiment très positif, aucune pause déclenchée")
                    news["processed"] = True
                    continue

            # --- Multiples sources et impact fort ---
            if n_sources >= thresholds["sources"] and impact > thresholds["impact"]:
                print("⚠️ Multiples sources + impact élevé -> PAUSE GLOBALE")
                self.global_cycles_remaining = 8
                news["processed"] = True
                self.last_event_time = datetime.now()
                self.last_event_news = news
                self.last_triggered_title = title
                triggered = True
                continue

            print("\nRecherche des mots-clés critiques:")
            for keyword, pause_cycles in self.CRITICAL_KEYWORDS.items():
                if re.search(rf"\b{re.escape(keyword)}\b", content):
                    print(f"⚠️ Mot-clé '{keyword}' trouvé!")
                    if title == self.last_triggered_title:
                        print("➡️ Même titre que précédent, skip")
                        continue
                    cycles = pause_cycles
                    if sentiment is not None and abs(sentiment) > 0.7:
                        cycles = int(cycles * 1.5)
                        print(f"Durée augmentée (sentiment fort): {cycles}")
                    elif sentiment is not None and abs(sentiment) < 0.3:
                        cycles = int(max(2, cycles * 0.5))
                        print(f"Durée réduite (sentiment faible): {cycles}")
                    news["processed"] = True
                    if symbols:
                        for sym in symbols:
                            if keyword in [
                                "regulation",
                                "lawsuit",
                                "investigation",
                                "ban",
                            ]:
                                self.buy_paused_pairs.add(sym)
                                self.pair_pauses[sym] = cycles
                                print(f"🔒 BUY PAUSE {cycles} cycles pour {sym}")
                            else:
                                self.pair_pauses[sym] = cycles
                                print(f"🔒 FULL PAUSE {cycles} cycles pour {sym}")
                    else:
                        self.global_cycles_remaining = cycles
                        print(f"🔒 PAUSE GLOBALE {cycles} cycles")
                    self.last_event_time = datetime.now()
                    self.last_event_news = news
                    self.last_triggered_title = title
                    if self.alert_callback:
                        self.alert_callback(keyword, news)
                    triggered = True
                    break
        print(
            f"\n[NEWSPAUSE DEBUG] Résultat final: {'⚠️ Pause activée' if triggered else '✅ Aucune pause nécessaire'}"
        )
        return triggered

    def reset_pauses(self, active_pauses):
        self.global_cycles_remaining = 0
        self.pair_pauses = {}
        if not active_pauses:
            return
        for pause in active_pauses:
            asset = pause.get("asset", "GLOBAL")
            cycles_left = int(pause.get("cycles_left", 0))
            if asset == "GLOBAL":
                self.global_cycles_remaining = cycles_left
            else:
                self.pair_pauses[asset] = cycles_left

    def smart_pause_update(self, bot):
        regime = getattr(bot, "regime", None)
        sentiment = None
        try:
            sentiment = bot.get_performance_metrics().get("sentiment", 0)
        except Exception:
            pass

        if regime == "TRENDING_UP":
            if self.global_cycles_remaining > 2:
                print("[SMART PAUSE] Marché haussier, réduction de la pause globale !")
                self.global_cycles_remaining = max(self.global_cycles_remaining // 2, 1)
        if sentiment is not None and sentiment > 0.5:
            print(
                "[SMART PAUSE] Sentiment news positif, réduction de la pause globale !"
            )
            self.global_cycles_remaining = max(self.global_cycles_remaining // 2, 1)
        for pair, cycles in list(self.pair_pauses.items()):
            vol = bot.calculate_volatility(bot.market_data.get(pair, {}).get("1h", {}))
            avg_vol = bot.calculate_volume_profile(
                bot.market_data.get(pair, {}).get("1h", {})
            ).get("strength", 1)
            if vol < 0.05 and avg_vol > 0.7:
                print(f"[SMART PAUSE] Volatilité/volume OK sur {pair}, pause réduite !")
                self.pair_pauses[pair] = max(cycles // 2, 1)
        for pair, cycles in list(self.pair_pauses.items()):
            market_data = bot.market_data.get(pair, {}).get("1h", {})
            if "close" in market_data and len(market_data["close"]) > 10:
                prices = market_data["close"][-10:]
                if prices[-1] > prices[0] * 1.07:
                    print(
                        f"[SMART PAUSE] Prix {pair} +7% pendant la pause, pause levée !"
                    )
                    self.pair_pauses[pair] = 0

    def safe_update_shared_data(
        self, new_fields: dict, data_file="src/shared_data.json"
    ):
        # Fusion profonde obligatoire !
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                elif isinstance(v, list) and k in d and isinstance(d[k], list):
                    if k == "pending_sales":
                        existing = {
                            item.get("symbol"): item
                            for item in d[k]
                            if isinstance(item, dict) and "symbol" in item
                        }
                        for new_item in v:
                            if isinstance(new_item, dict) and "symbol" in new_item:
                                existing[new_item["symbol"]] = new_item
                        d[k] = list(existing.values())
                    else:
                        d[k] = v
                else:
                    d[k] = v
            return d

        try:
            with open(data_file, "r") as f:
                shared_data = json.load(f)
        except Exception:
            backup_file = data_file + ".bak"
            if os.path.exists(backup_file):
                with open(backup_file, "r") as f:
                    shared_data = json.load(f)
            else:
                shared_data = {}
        if shared_data is None:
            print("[SAFE PATCH] shared_data.json corrompu, skip écriture !")
            return
        shared_data = deep_update(shared_data, new_fields)
        try:
            shutil.copyfile(data_file, data_file + ".bak")
        except Exception:
            pass
        with open(data_file, "w") as f:
            json.dump(shared_data, f, indent=4)

    def activate_pause(self, pause_decision):
        self.active_pauses.append(pause_decision)

    def should_pause(self, news_item, market_data):
        sentiment = news_item.get("sentiment", 0)
        impact = news_item.get("impact_score", 0)
        risk_class = news_item.get("risk_class", "")
        n_sources = news_item.get("n_sources", 1)
        symbol = news_item.get("symbols", ["GLOBAL"])[0]
        vol_before = market_data.get(symbol, {}).get("volatility", 0)
        vol_after = market_data.get(symbol, {}).get("volatility_post_news", vol_before)
        if risk_class in ["Réglementaire", "Hack"] and impact > 0.6:
            return {
                "type": "total",
                "reason": news_item.get("title"),
                "duration": 10,
                "pair": symbol,
            }
        if sentiment < -0.5 and vol_after > vol_before * 2:
            return {
                "type": "pair",
                "pair": symbol,
                "reason": news_item.get("title"),
                "duration": 5,
            }
        if n_sources >= 2 and impact > 0.5:
            return {
                "type": "total",
                "reason": news_item.get("title"),
                "duration": 10,
                "pair": symbol,
            }
        if "short squeeze" in news_item.get("title", "").lower():
            return {
                "type": "short_only",
                "reason": news_item.get("title"),
                "duration": 3,
                "pair": symbol,
            }
        return None

    def on_cycle_end(self):
        print(
            "[NEWSPAUSE] Avant decrement:",
            self.global_cycles_remaining,
            self.pair_pauses,
        )
        to_remove = []
        if self.global_cycles_remaining > 0:
            self.global_cycles_remaining -= 1
        for pair, cycles in list(self.pair_pauses.items()):
            if cycles > 0:
                self.pair_pauses[pair] = max(0, cycles - 1)
            if self.pair_pauses[pair] <= 0:
                to_remove.append(pair)
        for pair in to_remove:
            self.pair_pauses.pop(pair, None)
            self.buy_paused_pairs.discard(pair)
        try:
            current_pauses = self.get_active_pauses()
            pair_cycle_values = [
                v for v in self.pair_pauses.values() if isinstance(v, (int, float))
            ]
            max_remaining = max(
                [self.global_cycles_remaining] + pair_cycle_values, default=0
            )
            self.safe_update_shared_data(
                {
                    "active_pauses": current_pauses,
                    "pause_status": {
                        "global_remaining": self.global_cycles_remaining,
                        "pair_pauses": self.pair_pauses,
                        "max_remaining": max_remaining,
                    },
                }
            )
        except Exception as e:
            print(f"[ERROR] Erreur mise à jour pauses: {e}")
        print(
            "[NEWSPAUSE] Après decrement:",
            self.global_cycles_remaining,
            self.pair_pauses,
        )

    def get_last_event(self):
        return self.last_event_news

    def reset(self):
        self.global_cycles_remaining = 0
        self.pair_pauses.clear()
        self.buy_paused_pairs.clear()
        self.last_event_news = None
        self.last_event_time = None
        self.last_triggered_title = None

    def get_active_pauses(self):
        pauses = []
        pair_cycle_values = [
            v for v in self.pair_pauses.values() if isinstance(v, (int, float))
        ]
        max_cycles = max([self.global_cycles_remaining] + pair_cycle_values, default=0)
        for pair, cycles_left in self.pair_pauses.items():
            if isinstance(cycles_left, (int, float)) and cycles_left > 0:
                pause_type = "BUY" if pair in self.buy_paused_pairs else "FULL"
                last_news = getattr(self, "last_event_news", None)
                reason = (
                    last_news.get("title", "") if isinstance(last_news, dict) else ""
                )
                pauses.append(
                    {
                        "asset": pair,
                        "action": pause_type,
                        "cycles_left": cycles_left,
                        "type": pause_type,
                        "reason": reason,
                        "max_cycles": max_cycles,
                    }
                )
        if (
            isinstance(self.global_cycles_remaining, (int, float))
            and self.global_cycles_remaining > 0
        ):
            last_news = getattr(self, "last_event_news", None)
            reason = last_news.get("title", "") if isinstance(last_news, dict) else ""
            pauses.append(
                {
                    "asset": "GLOBAL",
                    "action": "ALL",
                    "cycles_left": self.global_cycles_remaining,
                    "type": "GLOBAL",
                    "reason": reason,
                    "max_cycles": max_cycles,
                }
            )
        if pauses:
            for pause in pauses:
                pause["total_remaining"] = max_cycles
        return pauses

    def analyze_market_conditions(self, price_data, volume_data):
        for symbol in price_data:
            volatility = self.calculate_rolling_volatility(price_data[symbol])
            volume_profile = self.analyze_volume_profile(volume_data[symbol])
            momentum = self.calculate_momentum_score(price_data[symbol])
            self.market_conditions[symbol] = {
                "volatility": volatility,
                "volume_profile": volume_profile,
                "momentum": momentum,
            }

    def should_enter_trade(self, symbol):
        conditions = self.market_conditions.get(symbol, {})
        if (
            conditions.get("volatility", 1) < self.volatility_thresholds["medium"]
            and conditions.get("volume_profile", 0) > 0.7
            and conditions.get("momentum", 0) > 0
        ):
            return True
        return False
