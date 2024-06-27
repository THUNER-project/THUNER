"""Set up logging."""

import logging
from thor.config import get_outputs_directory


def setup_logger(name, level=logging.DEBUG):
    """
    Function to set up a logger with the specified name, log file, and log level.

    Parameters
    ----------
    name : str
        Name of the logger.
    log_file : str
        File path to save the log file.
    level : int, optional
        Logging level (default is logging.INFO).

    Returns
    -------
    logger : logging.Logger
        The configured logger object.
    """

    # If a logger with this name already exists, return it
    if logging.getLogger(name).hasHandlers():
        return logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_dir = get_outputs_directory() / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / f"{name}.log"

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
