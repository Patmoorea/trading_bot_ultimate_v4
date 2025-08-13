class RiskManager:
    def __init__(self):
        self.params = {
            'max_drawdown': 0.05,
            'daily_stop_loss': 0.02
        }
    def check_risk(self, portfolio):
        drawdown = (portfolio.high - portfolio.current) / portfolio.high
        if drawdown > self.params['max_drawdown']:
            return 'REDUCE_POSITIONS'
        return 'NORMAL'
