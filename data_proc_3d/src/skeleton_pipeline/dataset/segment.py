"""Splits one {"metadata", "features", "labels"} .pt take (as built by
app/build_training_pairs.py or app/augment_dataset.py) into fixed-length frame
segments -- the 3D counterpart of data_proc_2d/app/segment_data.py. That
script's slicing logic is dtype-agnostic (works on any tensor/array under
"features"/"labels", regardless of what pipeline built them), so this is a
close port rather than a reimplementation-from-scratch."""
import re
from pathlib import Path
from typing import Iterator


def iter_segments(data: dict, segment_size: int) -> Iterator[tuple[int, dict]]:
    """Yields (segment_index, segment_data) for each segment_size-frame
    chunk of *data* (a loaded take dict with "metadata"/"features"/"labels"
    keys). metadata["total_frames"] drives how many segments are produced;
    every array/tensor under "features" and "labels" is sliced [start:end]
    along its leading (frame) axis."""
    metadata = data.get("metadata", {})
    features = data.get("features", {})
    labels = data.get("labels", {})
    total_frames = metadata.get("total_frames", 0)

    for segment_index, start in enumerate(range(0, total_frames, segment_size)):
        end = start + segment_size
        segment_metadata = dict(metadata)
        segment_metadata["segment_index"] = segment_index
        segment_metadata["segment_size"] = segment_size
        segment_metadata["frame_start"] = start
        segment_metadata["frame_end"] = end

        segment_data = {
            "metadata": segment_metadata,
            "features": {key: _materialize_slice(value, start, end) for key, value in features.items()},
            "labels": {key: _materialize_slice(value, start, end) for key, value in labels.items()},
        }
        yield segment_index, segment_data


def _materialize_slice(value, start: int, end: int):
    sliced = value[start:end]
    if hasattr(sliced, "clone"):
        return sliced.clone()
    if hasattr(sliced, "copy"):
        try:
            return sliced.copy()
        except TypeError:
            pass
    return sliced


def format_segment_file_name(input_file_name: str, segment_index: int) -> str:
    """"features__cam-04_uid-01_take-01.pt" -> "features__cam-04_uid-01_take-01_seg-00.pt"
    (also matches the "_aug-NN" suffix augment_dataset.py adds)."""
    pattern = r"^(features__cam-\d{2}_uid-\d{2}_take-\d{2}(?:_aug-\d{2})?)\.pt$"
    replacement = rf"\1_seg-{segment_index:02}.pt"
    return re.sub(pattern, replacement, input_file_name)
