def validate_config(config):
    """Valide la configuration de base"""
    required_keys = {'api_key', 'api_secret', 'symbols', 'timeframes'}
    return all(key in config for key in required_keys)
