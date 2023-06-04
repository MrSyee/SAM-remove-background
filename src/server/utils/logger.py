import logging
import logging.config
import os


def get_logger(dir_name: str = "logs", config: str = "logging.conf") -> logging.Logger:
    """Get logger."""
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logging.config.fileConfig(config)
    logger = logging.getLogger()
    return logger
