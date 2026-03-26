import logging
import sys
from typing import Any, Dict, Optional


# ------------------------------------------------------------------
# Custom Levels
# ------------------------------------------------------------------

SUCCESS_LEVEL = 25
PINK_LEVEL = 35

logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')
logging.addLevelName(PINK_LEVEL, 'PINK')


def success(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


def pink(self: logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(PINK_LEVEL):
        self._log(PINK_LEVEL, message, args, **kwargs)


logging.Logger.success = success  # type: ignore[attr-defined]
logging.Logger.pink = pink  # type: ignore[attr-defined]


# ------------------------------------------------------------------
# Colored Formatter
# ------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    RESET = '\033[0m'

    COLORS = {
        logging.DEBUG: RESET,
        logging.INFO: RESET,
        SUCCESS_LEVEL: '\033[32m',
        logging.WARNING: '\033[33m',
        logging.ERROR: '\033[31m',
        logging.CRITICAL: '\033[41m',
        PINK_LEVEL: '\033[1;35m',
    }

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)

        color = self.COLORS.get(record.levelno)

        if color:
            return f'{color}{message}{self.RESET}'

        return message


# ------------------------------------------------------------------
# Configure logging
# ------------------------------------------------------------------


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)

    formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s => %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    root.handlers.clear()
    root.addHandler(handler)


# ------------------------------------------------------------------
# Context Logger
# ------------------------------------------------------------------


class ContextLogger:
    def __init__(self, logger: logging.Logger, **base_context: Any) -> None:
        self._logger = logger
        self._base_context: Dict[str, Any] = dict(base_context)

    def bind(self, **extra_context: Any) -> 'ContextLogger':
        merged = dict(self._base_context)
        merged.update(extra_context)

        return ContextLogger(self._logger, **merged)

    def _context(self, **extra: Any) -> Dict[str, Any]:
        context = dict(self._base_context)
        context.update(extra)

        return context

    def _log(self, level: int, message: str, *args: Any, **extra: Any) -> None:
        context = self._context(**extra)

        if context:
            self._logger.log(level, '%s | context=%s', message, context, *args)
        else:
            self._logger.log(level, message, *args)

    # --------------------------------------------------------------

    def debug(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(logging.DEBUG, message, *args, **extra)

    def info(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(logging.INFO, message, *args, **extra)

    def success(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(SUCCESS_LEVEL, message, *args, **extra)

    def warning(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(logging.WARNING, message, *args, **extra)

    def error(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(logging.ERROR, message, *args, **extra)

    def critical(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(logging.CRITICAL, message, *args, **extra)

    def pink(self, message: str, *args: Any, **extra: Any) -> None:
        self._log(PINK_LEVEL, message, *args, **extra)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def get_logger(name: Optional[str] = None, **context: Any) -> ContextLogger:
    base_logger = logging.getLogger(name)

    return ContextLogger(base_logger, **context)
