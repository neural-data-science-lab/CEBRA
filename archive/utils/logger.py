"""
Logger Utility

Provides a standardized logger instance with consistent formatting
and avoids duplicate handlers across the project.

Required packages:
    - logging

Author:
Created: 07.08.2025
Last updated: 07.08.2025
"""

# --------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------

import logging

# --------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------

def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Create and return a logger with INFO level and a standardized format.
    Ensures no duplicate handlers are added.

    Args:
        name (str): Logger name. Defaults to the module's __name__.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
