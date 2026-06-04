"""Unit tests for thuner.log.

setup_logger's main (handler-configuring) branch is normally not exercised under
pytest: pytest attaches a handler to the *root* logger for log capture, and
``logging.getLogger(name).hasHandlers()`` walks up to the root, so the early-exit
return is taken for every named logger. Setting ``propagate = False`` below stops
that walk so the configuring branch actually runs.
"""

import logging

import thuner.log as log


def test_setup_logger_returns_existing_when_handlers_present():
    name = "thuner._existing_logger_test"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    handler = logging.NullHandler()
    logger.addHandler(handler)
    try:
        result = log.setup_logger(name)
        assert result is logger
        assert result.handlers == [handler]  # no extra handlers added
    finally:
        logger.handlers.clear()


def test_setup_logger_configures_handlers(tmp_path, monkeypatch):
    monkeypatch.setattr(log, "get_outputs_directory", lambda: tmp_path)
    name = "thuner._fresh_logger_test"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    try:
        result = log.setup_logger(name)
        assert len(result.handlers) == 2
        assert any(isinstance(h, logging.FileHandler) for h in result.handlers)
        assert (tmp_path / "log" / f"{name}.log").exists()
    finally:
        logger.handlers.clear()
        logger.propagate = True


def test_logging_listener_starts_and_stops():
    with log.logging_listener():
        assert log.listener is not None
    assert log.listener is None
