import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

class LiveDashboard:
    """Tableau de bord en temps réel pour le suivi des performances"""
    
    def __init__(self, refresh_interval=2000):
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 8))
        self.refresh_interval = refresh_interval
        self.animation = None
        
    def setup_layout(self):
        """Configurer la mise en page du dashboard"""
        plt.tight_layout()
        self.axes[0].set_title('Performance du Portefeuille')
        self.axes[1].set_title('Métriques de Risque')
        
    def update_plots(self, data):
        """Mettre à jour les graphiques avec les nouvelles données"""
        # Effacer les axes précédents
        for ax in self.axes:
            ax.clear()
            
        # Plot 1: Courbe de performance
        data['equity'].plot(ax=self.axes[0])
        self.axes[0].set_ylabel('Valeur')
        
        # Plot 2: Métriques de risque
        data['drawdown'].plot(ax=self.axes[1], color='red')
        self.axes[1].set_ylabel('Drawdown (%)')
        
        self.setup_layout()
    
    def start(self, data_callback):
        """Démarrer l'animation en temps réel"""
        self.setup_layout()
        self.animation = FuncAnimation(
            self.fig,
            lambda i: self.update_plots(data_callback()),
            interval=self.refresh_interval
        )
        plt.show()