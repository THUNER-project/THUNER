"""Set up logging."""

import logging
import logging.handlers
import multiprocessing
from contextlib import contextmanager
from thuner.config import get_outputs_directory


log_queue = multiprocessing.Queue(-1)
listener = None


def setup_logger(name, level=logging.INFO):
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

    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_all_loggers():
    """Get a list of all currently active loggers."""
    logger_dict = logging.Logger.manager.loggerDict
    active_loggers = [
        logging.getLogger(name)
        for name in logger_dict
        if isinstance(logger_dict[name], logging.Logger)
    ]
    return active_loggers


def start_listener():
    """Start the logging listener process."""
    global listener
    root_logger = logging.getLogger()
    listener = logging.handlers.QueueListener(log_queue, *root_logger.handlers)
    listener.start()


def stop_listener():
    """Stop the logging listener process."""
    global listener
    if listener:
        listener.stop()
        listener = None


@contextmanager
def logging_listener():
    """Context manager to start and stop the logging listener."""
    start_listener()
    try:
        yield
    finally:
        stop_listener()
