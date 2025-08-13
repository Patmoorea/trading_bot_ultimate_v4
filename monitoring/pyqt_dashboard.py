from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
class TradingDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('Trading Bot Dashboard')
        self.setGeometry(100, 100, 1200, 800)
