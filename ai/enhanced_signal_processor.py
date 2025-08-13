class EnhancedSignalProcessor:
    def __init__(self):
        self.model = self._load_enhanced_model()
        self.calibrator = SignalCalibrator()
        self.validator = SignalValidator()
    def process_signal(self, data):
        """Traitement amélioré des signaux avec validation multi-niveau"""
        # Pré-traitement avec filtrage du bruit
        clean_data = self.preprocessor.denoise(data)
        # Génération signal avec modèle amélioré
        raw_signal = self.model.predict(clean_data)
        # Calibration dynamique
        calibrated_signal = self.calibrator.calibrate(
            raw_signal, 
            market_context=self.get_market_context()
        )
        # Validation multi-niveau
        if self.validator.validate(calibrated_signal):
            confidence = self.calculate_confidence(calibrated_signal)
            if confidence > 0.85:  # Seuil de confiance augmenté
                return calibrated_signal
        return None
