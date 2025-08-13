class EnhancedRiskManager:
    def __init__(self):
        # Limites et seuils
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.position_limits = {
            "max_per_trade": 0.05,  # 5% max par trade
            "max_total_exposure": 0.25,  # 25% max exposition totale
        }
        self.min_confidence = 0.8  # 80% confiance minimum
        self.correlation_threshold = 0.7  # Seuil corrélation max

        # Seuils de validation
        self.validation_thresholds = {
            "technical": 0.3,  # Score technique minimum
            "momentum": 0.2,  # Score momentum minimum
            "orderflow": 0.2,  # Score orderflow minimum
            "liquidity": 0.7,  # Seuil liquidité maximum
            "pressure": 0.8,  # Seuil pression marché maximum
        }
        # Nouveaux seuils techniques
        self.tech_thresholds = {
            "trend_score": 0.7,
            "volume_score": 0.6,
            "momentum_score": 0.6,
            "sr_score": 0.7,
        }

    def validate_trade(self, signals):
        """Validation complète incluant les signaux techniques"""
        try:
            if not signals or not isinstance(signals, dict):
                print("[RISK] Signaux invalides")
                return False

            # Extraction et validation des composantes principales
            technical = signals.get("technical", {})
            momentum = signals.get("momentum", {})
            orderflow = signals.get("orderflow", {})

            if not all([technical, momentum, orderflow]):
                print("[RISK] Composantes de signal manquantes")
                return False

            # Scores techniques
            tech_score = float(technical.get("score", 0))
            tech_factors = technical.get("factors", 0)

            # Validation technique améliorée
            if abs(tech_score) < 0.3 or tech_factors < 2:
                print(f"[RISK] Score technique insuffisant: {tech_score:.2f}")
                return False

            # Validation momentum
            momentum_score = float(momentum.get("score", 0))
            if abs(momentum_score) < 0.2:
                print(f"[RISK] Momentum insuffisant: {momentum_score:.2f}")
                return False

            # Validation orderflow
            flow_score = float(orderflow.get("score", 0))
            if abs(flow_score) < 0.2:
                print(f"[RISK] Orderflow insuffisant: {flow_score:.2f}")
                return False

            # Score global
            weights = {"technical": 0.4, "momentum": 0.3, "orderflow": 0.3}
            total_score = (
                tech_score * weights["technical"]
                + momentum_score * weights["momentum"]
                + flow_score * weights["orderflow"]
            )

            # Validation finale
            if abs(total_score) < 0.25:
                print(f"[RISK] Score global insuffisant: {total_score:.2f}")
                return False

            print(f"[RISK] ✅ Trade validé - Score: {total_score:.2f}")
            return True

        except Exception as e:
            print(f"[RISK] Erreur validation: {e}")
            return False

    def calculate_position_size(self, equity, confidence, volatility, correlation):
        """Calcul intelligent de la taille de position"""
        try:
            if confidence < self.min_confidence:
                print(f"[RISK] Confiance insuffisante: {confidence:.2f}")
                return 0

            base_size = equity * self.position_limits["max_per_trade"]

            # Ajustements
            vol_adj = max(0.3, 1 - (volatility * 2))
            corr_adj = max(0.3, 1 - correlation)

            # Calcul final
            size = base_size * vol_adj * corr_adj
            final_size = min(size, equity * self.position_limits["max_per_trade"])

            print(f"[RISK] Taille calculée: {final_size:.2f} USDC")
            return float(final_size)

        except Exception as e:
            print(f"[RISK] Erreur calcul position: {e}")
            return 0

    def check_exposure_limit(self, current_positions, new_position_size):
        """Vérification des limites d'exposition"""
        try:
            total_exposure = sum(
                float(pos.get("size", 0)) for pos in current_positions.values()
            )
            new_total = total_exposure + float(new_position_size)
            is_valid = new_total <= self.position_limits["max_total_exposure"]

            print(f"[RISK] Exposition totale: {new_total:.2%}")
            return is_valid

        except Exception as e:
            print(f"[RISK] Erreur vérification exposition: {e}")
            return False

    def calculate_drawdown(self, equity_curve):
        """Calcul du drawdown actuel"""
        try:
            if not equity_curve:
                return 0

            peak = max(equity_curve)
            current = equity_curve[-1]

            if peak == 0:
                return 0

            drawdown = (current - peak) / peak
            print(f"[RISK] Drawdown actuel: {drawdown:.2%}")

            return abs(drawdown)

        except Exception as e:
            print(f"[RISK] Erreur calcul drawdown: {e}")
            return 0

    def adjust_for_market_conditions(self, base_size, market_regime):
        """Ajustement selon conditions de marché"""
        try:
            # Ajustements par régime
            regime_multipliers = {
                "TRENDING_UP": 1.0,  # Normal en tendance haussière
                "TRENDING_DOWN": 0.7,  # Réduction en tendance baissière
                "RANGING": 0.8,  # Réduction en range
                "VOLATILE": 0.5,  # Forte réduction en volatilité
            }

            multiplier = regime_multipliers.get(market_regime, 0.7)
            adjusted_size = base_size * multiplier

            print(f"[RISK] Ajustement régime {market_regime}: {multiplier:.2f}x")
            return adjusted_size

        except Exception as e:
            print(f"[RISK] Erreur ajustement marché: {e}")
            return base_size * 0.7  # Réduction par défaut
