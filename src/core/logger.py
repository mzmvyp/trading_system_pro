"""
Centralized logging configuration for Agno Trading Bot
Provides structured logging with file rotation and console output

Logs separados:
- logs/trading_bot.log  → log principal (trading, agent, etc.)
- logs/ml.log           → ML, LSTM, online learning
- logs/backtest.log     → backtesting, otimização
- Console               → só trading (ML/backtest ficam só em arquivo)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Módulos que vão pra logs separados (sem console)
_ML_PREFIXES = ("src.ml",)
_BACKTEST_PREFIXES = ("src.backtesting",)


def _is_ml_logger(name: str) -> bool:
    return any(name.startswith(p) for p in _ML_PREFIXES)


def _is_backtest_logger(name: str) -> bool:
    return any(name.startswith(p) for p in _BACKTEST_PREFIXES)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup a logger with both file and console handlers

    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path. If None, uses logs/{name}.log
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum log file size before rotation (default 10MB)
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Determine log file path based on module
    if log_file is None:
        if _is_ml_logger(name):
            log_file = log_dir / "ml.log"
        elif _is_backtest_logger(name):
            log_file = log_dir / "backtest.log"
        else:
            log_file = log_dir / f"{name.replace('.', '_')}.log"
    else:
        log_file = Path(log_file)

    # Format for log messages
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler - só para módulos de TRADING (ML/backtest ficam só em arquivo)
    if not _is_ml_logger(name) and not _is_backtest_logger(name):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Evitar propagação para o root logger (previne duplicação)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)


# Create main application logger
main_logger = setup_logger("agno_trading_bot", log_file="logs/trading_bot.log")
