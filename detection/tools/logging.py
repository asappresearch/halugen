import logging

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def apply_log_level(log_level: str) -> None:
    level = LOG_LEVELS.get(log_level.upper(), "")
    if not level:
        print(f"Unrecognized log_level: {log_level}")
        return

    # Apply to the default logger
    logging.basicConfig(level=level)

