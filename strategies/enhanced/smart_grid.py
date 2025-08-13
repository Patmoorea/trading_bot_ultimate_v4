import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
@dataclass
class GridConfig:
    grid_levels: int = 10
    grid_spacing: float = 0.01
    position_size: float = 0.1
    max_positions: int = 5
    take_profit: float = 0.03
    stop_loss: float = 0.02
    use_quantum: bool = True
class SmartGridStrategy:
    def __init__(self, config: GridConfig = GridConfig()):
        self.config = config
        self.positions = []
        self.grids = []
        self.quantum_state = None
    def calculate_grids(self, price: float) -> List[Dict]:
        """Calcule les niveaux de grid"""
        grids = []
        for i in range(-self.config.grid_levels, self.config.grid_levels + 1):
            level_price = price * (1 + i * self.config.grid_spacing)
            grids.append({
                "price": level_price,
                "type": "buy" if i < 0 else "sell",
                "size": self.config.position_size,
                "active": True
            })
        return grids
    def update_quantum_state(self, market_data: pd.DataFrame) -> None:
        """Met à jour l'état quantique pour les décisions"""
        if self.config.use_quantum:
            from src.quantum.core.quantum_detector import QuantumPatternDetector
            detector = QuantumPatternDetector()
            self.quantum_state = detector.detect_patterns(
                market_data[["close", "volume"]].values
            )
    def adjust_grids(self, price: float) -> None:
        """Ajuste les grids en fonction du prix et de l'état quantique"""
        # Ajustement basé sur l'état quantique
        if self.quantum_state and "patterns" in self.quantum_state:
            patterns = self.quantum_state["patterns"]
            for pattern in patterns:
                if pattern == "bullish_divergence":
                    self._shift_grids_up(price)
                elif pattern == "bearish_divergence":
                    self._shift_grids_down(price)
        # Nettoyage des positions fermées
        self.positions = [p for p in self.positions if p["active"]]
        # Mise à jour des grids actifs
        self.grids = self.calculate_grids(price)
    def _shift_grids_up(self, price: float) -> None:
        """Déplace les grids vers le haut"""
        shift_factor = 1 + self.config.grid_spacing
        self.grids = [{
            **grid,
            "price": grid["price"] * shift_factor
        } for grid in self.grids]
    def _shift_grids_down(self, price: float) -> None:
        """Déplace les grids vers le bas"""
        shift_factor = 1 - self.config.grid_spacing
        self.grids = [{
            **grid,
            "price": grid["price"] * shift_factor
        } for grid in self.grids]
    def check_exits(self, price: float) -> List[Dict]:
        """Vérifie les conditions de sortie"""
        exits = []
        for position in self.positions:
            if not position["active"]:
                continue
            entry_price = position["price"]
            if position["type"] == "buy":
                profit_price = entry_price * (1 + self.config.take_profit)
                loss_price = entry_price * (1 - self.config.stop_loss)
                if price >= profit_price:
                    exits.append({
                        "type": "tp_sell",
                        "position": position,
                        "price": price
                    })
                elif price <= loss_price:
                    exits.append({
                        "type": "sl_sell",
                        "position": position,
                        "price": price
                    })
            else:  # position["type"] == "sell"
                profit_price = entry_price * (1 - self.config.take_profit)
                loss_price = entry_price * (1 + self.config.stop_loss)
                if price <= profit_price:
                    exits.append({
                        "type": "tp_buy",
                        "position": position,
                        "price": price
                    })
                elif price >= loss_price:
                    exits.append({
                        "type": "sl_buy",
                        "position": position,
                        "price": price
                    })
        return exits
    def get_signals(self, price: float) -> List[Dict]:
        """Obtient les signaux de trading"""
        signals = []
        # Vérifie d'abord les sorties
        exits = self.check_exits(price)
        for exit_signal in exits:
            signals.append(exit_signal)
            exit_signal["position"]["active"] = False
        # Vérifie ensuite les entrées sur les grids
        if len(self.positions) < self.config.max_positions:
            for grid in self.grids:
                if not grid["active"]:
                    continue
                if grid["type"] == "buy" and price <= grid["price"]:
                    signals.append({
                        "type": "grid_buy",
                        "price": grid["price"],
                        "size": grid["size"]
                    })
                    grid["active"] = False
                    self.positions.append({
                        "type": "buy",
                        "price": grid["price"],
                        "size": grid["size"],
                        "active": True
                    })
                elif grid["type"] == "sell" and price >= grid["price"]:
                    signals.append({
                        "type": "grid_sell",
                        "price": grid["price"],
                        "size": grid["size"]
                    })
                    grid["active"] = False
                    self.positions.append({
                        "type": "sell",
                        "price": grid["price"],
                        "size": grid["size"],
                        "active": True
                    })
        return signals
