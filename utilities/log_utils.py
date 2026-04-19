
import logging
import sys


_FORMATTER = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def setup_logger(message: str = "") -> logging.Logger:
    logger = logging.getLogger(message)
    if not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(_FORMATTER)
        # Force UTF-8 on the underlying stream to handle non-ASCII paths
        # on Windows consoles whose default codepage can't encode them.
        if hasattr(stream_handler.stream, "reconfigure"):
            try:
                stream_handler.stream.reconfigure(encoding="utf-8", errors="replace")
            except ValueError:
                pass
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def save_log_to_file(logger: logging.Logger, file_path: str):
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)