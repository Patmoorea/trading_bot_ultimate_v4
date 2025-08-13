LOG_FORMAT = "%(asctime)s - EXEC_TIME: %(exec_time).3fms"
def log_error(message: str, level: str = "ERROR") -> None:
    """Nouvelle fonction de log des erreurs
    Args:
        message: Message d'erreur à logger
        level: Niveau de log (ERROR, WARNING, INFO)
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if level.upper() == "ERROR":
        logger.error(message)
    elif level.upper() == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)
