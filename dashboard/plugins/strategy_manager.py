import streamlit as st
import importlib
import inspect
from typing import Dict, List, Type, Optional
from pathlib import Path
import logging
class StrategyManager:
    def __init__(self, strategies_path: str = "src/strategies"):
        self.logger = logging.getLogger(__name__)
        self.strategies_path = Path(strategies_path)
        self.available_strategies = self._load_strategies()
    def _load_strategies(self) -> Dict[str, Type]:
        """Charge dynamiquement les stratégies disponibles"""
        strategies = {}
        try:
            # Parcourir les fichiers Python dans le dossier strategies
            for strategy_file in self.strategies_path.glob("**/*.py"):
                if strategy_file.name.startswith("_"):
                    continue
                try:
                    # Importer le module
                    module_path = str(strategy_file.relative_to(Path.cwd()))[:-3].replace("/", ".")
                    module = importlib.import_module(module_path)
                    # Trouver les classes de stratégie
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            hasattr(obj, "run") and 
                            hasattr(obj, "name")):
                            strategies[obj.name] = obj
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement de {strategy_file}: {e}")
        except Exception as e:
            self.logger.error(f"Erreur lors du scan des stratégies: {e}")
        return strategies
    def render_strategy_selector(self) -> Dict[str, bool]:
        """Affiche le sélecteur de stratégies"""
        st.subheader("🎯 Stratégies")
        selected_strategies = {}
        for name, strategy_class in self.available_strategies.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.checkbox(
                    name,
                    help=strategy_class.__doc__
                )
            with col2:
                st.button(
                    "⚙️",
                    key=f"config_{name}",
                    help="Configurer la stratégie"
                )
            selected_strategies[name] = selected
            if selected and hasattr(strategy_class, "render_config"):
                with st.expander("Configuration"):
                    strategy_class.render_config()
        return selected_strategies
    def get_active_strategies(self) -> List[Type]:
        """Retourne les stratégies actives"""
        active = []
        if "selected_strategies" in st.session_state:
            selected = st.session_state.selected_strategies
            active = [
                self.available_strategies[name]
                for name, is_active in selected.items()
                if is_active
            ]
        return active
