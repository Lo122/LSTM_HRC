"""Small file-I/O + logging helpers for the training-data pipeline
(app/build_training_pairs.py) -- mirrors the conventions of data_proc_2d's
file_io_utils.py / utilities.file_io.py / utilities.log_utils.py (save_torch/
load_torch/iter_files, setup_logger/save_log_to_file), reimplemented here
self-contained (no cross-project `utilities` import) since data_proc_3d/app
deliberately runs in its own minimal venv -- see generate_lstm_training_data.py's
module docstring on why data_proc_2d/data_proc_3d don't share a Python env.
"""
import json
import logging
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_torch(data: Any, path: Path, logger: logging.Logger | None = None) -> None:
    import torch
    ensure_parent_dir(path)
    torch.save(data, path)
    if logger:
        logger.info("Saved torch object to %s", path)


def load_torch(path: Path, logger: logging.Logger | None = None, map_location: Any = "cpu") -> Any:
    import torch
    data = torch.load(path, map_location=map_location, weights_only=False)
    if logger:
        logger.info("Loaded torch object from %s", path)
    return data


def load_json(path: Path, logger: logging.Logger | None = None) -> dict | None:
    path = Path(path)
    if not path.exists():
        if logger:
            logger.warning("SKIP: file not found -> %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if logger:
        logger.info("Loaded JSON from %s", path)
    return data


def save_json(data: Any, path: Path, logger: logging.Logger | None = None) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if logger:
        logger.info("Saved JSON to %s", path)


def iter_files(root: Path, extension: str = ".pt"):
    """Yield all files with the given extension recursively under *root*."""
    for file in sorted(Path(root).rglob(f"*{extension}")):
        yield file


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def save_log_to_file(logger: logging.Logger, path: str) -> None:
    """Attach a FileHandler to *logger* so its output is also written to
    *path* -- mirrors utilities.log_utils.save_log_to_file's call signature
    (used as `log_utils.save_log_to_file(logger, str(folder_root / "logs/..."))`
    throughout data_proc_2d/app)."""
    log_path = Path(path)
    ensure_parent_dir(log_path)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
