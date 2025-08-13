class Config:
    # Optimisation
    USE_NUMBA = True  # Désactivé temporairement pour les tests
    USE_GPU = False
    # Paramètres de trading
    MAX_DRAWDOWN = 0.05
    DAILY_STOP_LOSS = 0.02
    # Mode debug
    DEBUG = True
    TEST_MODE = True
    # Logging
    LOG_LEVEL = 'DEBUG'
