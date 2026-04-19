import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
import re

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utilities import file_io


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _import_torch():
    """Import torch lazily to avoid hard crashes during module import."""
    try:
        import torch  # type: ignore

        return torch
    except OSError as exc:
        raise RuntimeError(
            "PyTorch could not load native DLLs. This usually means an incompatible "
            "Python/PyTorch build on Windows. Use Python 3.10-3.12 and reinstall torch."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the active environment. Install torch and retry."
        ) from exc


# pytorch's torch.save and torch.load functions with logging
def save_torch(data: Any, path: str, logger: logging.Logger | None = None) -> None:
    file_io.ensure_parent_dir(path)
    torch = _import_torch()
    torch.save(data, path)
    if logger:
        logger.info("Saved torch object to %s", path)


def load_torch(path: str, logger: logging.Logger | None = None, map_location: Any = "cpu") -> Any:
    torch = _import_torch()
    data = torch.load(path, map_location=map_location)
    if logger:
        logger.info("Loaded torch object from %s", path)
    return data


def load_step_ids_from_json(
    path: Path,
    logger: logging.Logger | None = None,
) -> tuple[list[dict[str, int | float]], str | None]:
    """Load step markers plus the referenced video path from a JSON label file."""
    try:
        data = file_io.load_json(str(path), logger=logger)
        step_markers = data.get("step_markers")
        video_file_path = data.get("video_path", "unknown")
        video_file_name = os.path.basename(video_file_path)
        fps = data.get("fps")

        label_info = []
        for step_marker in step_markers:
            step_id = step_marker.get("step_id")
            timestamp = step_marker.get("timestamp")
            if logger and step_id is not None and timestamp is not None:
                logger.info("Loaded step marker: step_id=%s, timestamp=%s", step_id, timestamp)

            frame = timestamp * fps
            raw = {"step_id": step_id, "timestamp": timestamp, "frame": frame}
            label_info.append(raw)

        return label_info, video_file_name

    except Exception as e:
        if logger:
            logger.error("Failed to load step IDs from JSON at %s: %s", path, e)
        return [], None


def load_elan_label_data(file_path: Path, config: dict, logger: logging.Logger | None = None) -> tuple[dict | None, str | None]:
    """Load ELAN .csv label data from *file_path*.

    Returns the parsed data as a dictionary, or ``None`` on any failure.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        description = file.readline().strip().strip('"')

    if logger:
        logger.info("Description: %s", description)
    meta_data = _extract_from_description(description, logger=logger)

    video_file_path = meta_data.get("file_url", "unknown").replace("file:///", "")
    video_file_name = os.path.basename(video_file_path)

    df = pd.read_csv(file_path, skiprows=1)

    df["Begin Time - ss.msec"] = pd.to_numeric(df["Begin Time - ss.msec"], errors="coerce")
    df["End Time - ss.msec"] = pd.to_numeric(df["End Time - ss.msec"], errors="coerce")
    df["Duration - ss.msec"] = pd.to_numeric(df["Duration - ss.msec"], errors="coerce")

    df["Begin Time - Frame"] = _time_to_frame(df, "Begin Time - ss.msec", meta_data.get("ms_per_sample", 1))
    df["End Time - Frame"] = _time_to_frame(df, "End Time - ss.msec", meta_data.get("ms_per_sample", 1))
    df["Duration - Frame"] = _time_to_frame(df, "Duration - ss.msec", meta_data.get("ms_per_sample", 1))
    
    for index, row in df.iterrows():
        for label in config.keys():
            try:
                if int(row[label]) == 1:
                    if logger:
                        logger.info("Found label '%s' at row %d (timestamp: %s)", label, index, row._2)
                df.loc[index, 'step_id'] = config[label]
                break
            except ValueError:
                if logger:
                    logger.warning("Invalid value for label '%s' at row %d: %s", label, index, row[label])

    columns = ["step_id", "Begin Time - ss.msec", "Begin Time - Frame", "End Time - Frame"]
    renamed_columns = {
        "Begin Time - ss.msec": "timestamp",
        "Begin Time - Frame": "start_frame",
        "End Time - Frame": "end_frame",
    }
    df_output = df[columns].copy()
    df_output.rename(columns=renamed_columns, inplace=True)
    df_output['step_id'] = df_output['step_id'].astype(int)
    output_dict = df_output.to_dict(orient="records")

    return output_dict, video_file_name


def _time_to_frame(df, column: str, ms_per_sample: float) -> int | None:
    """Convert time in seconds to frame number."""
    if column not in df:
        return None

    return df[column].apply(lambda x: int(round(x * 1000 / ms_per_sample)) if pd.notnull(x) else None)


def _extract_from_description(text: str, logger: logging.Logger | None = None) -> dict:
    """Extract key-value pairs from the description line of an ELAN .csv file."""
    pattern = re.compile(
        r'^"?#(?P<file_url>file:///.*?)\s+--\s+'
        r'offset:\s+(?P<offset>\d+(?:\.\d+)?)\s*,\s*'
        r'duration:\s+(?P<duration_hms>\d{2}:\d{2}:\d{2}\.\d+)\s*/\s*'
        r'(?P<duration_sec>\d+(?:\.\d+)?)\s*/\s*'
        r'(?P<duration_ms>\d+)\s*,\s*'
        r'ms per sample:\s+(?P<ms_per_sample>\d+(?:\.\d+)?)"?$'
    )

    match = pattern.match(text)

    if match:
        values = match.groupdict()

        values["offset"] = float(values["offset"])
        values["duration_sec"] = float(values["duration_sec"])
        values["duration_ms"] = int(values["duration_ms"])
        values["ms_per_sample"] = float(values["ms_per_sample"])
        if logger:
            logger.info("Extracted values from description: %s", values)
        return values

    if logger:
        logger.warning("No match found for description: %s", text)
    return {}



def modify_file_name(rel_path: Path, prefix: str = "features") -> Path:
    stem = rel_path.stem
    suffix_after_separator = stem.split("__", 1)[1] if "__" in stem else stem
    new_stem = f"{prefix}__{suffix_after_separator}"

    return rel_path.with_name(new_stem + rel_path.suffix)


if __name__ == "__main__":
    root_path = Path(r"G:\My Drive\University of Stuttgart\ITECH_Thesis\ELAN")
    file_path = root_path / "cam1_B1.csv"
    output_dict, video_file_name = load_elan_label_data(file_path)
    print(json.dumps(output_dict, indent=2, ensure_ascii=False))
    print("Referenced video file:", video_file_name)
