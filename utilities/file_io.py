import json
import logging
import os
from typing import Any



# JSON functions
def save_json(data: Any, path: str, logger: logging.Logger | None = None, indent: int = 2) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
    if logger:
        logger.info("Saved JSON to %s", path)


def load_json(path: str, logger: logging.Logger | None = None) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if logger:
        logger.info("Loaded JSON from %s", path)
    return data

# helper function to ensure parent directory exists before saving files
def ensure_parent_dir(path: str) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def pad_id(value: str) -> str:
    return f"{int(value):02d}"

