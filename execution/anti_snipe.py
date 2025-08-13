class AntiSnipeProtection:
    def __init__(self):
        self.last_order_time = None
    def check_snipe_risk(self, order):
        # Implémentation de la détection
        if self.last_order_time and (time.time() - self.last_order_time) < 0.1:
            return True
        return False
