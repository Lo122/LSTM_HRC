"""Debug helper for {"metadata", "features", "labels"} .pt takes: loads
them into pandas DataFrames and pauses in the VS Code debugger with all
of them in scope, to browse in the Variables pane / Data Viewer instead of
printing them to the terminal.

Run from the Run and Debug panel with one of the .vscode/launch.json
configurations:
  - "data_analysis.py: one take" -- asks for a .pt path, pauses with (in
    main()'s Locals):
      metadata          raw metadata dict
      features_df       (T, n_features) one column per feature, index = frame
      labels_df         (T, n_labels) *_vector labels split into _0 .. _6
      take_df           features_df + labels_df side by side, e.g. for
                        take_df[take_df.step_id == 3] in the Debug Console
      panel_dfs         {panel_key: that panel's slice of features_df}
      feature_stats_df  describe() per feature + panel + n_nonfinite
      label_stats_df    describe() per float label + n_nonfinite
      label_counts_df   frames per class id (-1 = no annotation) per int label
      missing_panel_columns  panel_columns names absent from features_df
  - "data_analysis.py: folder overview" -- asks for a dataset folder, pauses
    with overview_df (one row per .pt).
Right-click a DataFrame in the Variables pane -> "View Value in Data Viewer"
to sort/filter it. The pause only happens under the debugger; a plain
`uv run python data_analysis.py [--pt ... | --in-dir ...]` just prints a
short summary and exits.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from skeleton_pipeline.dataset import io_utils, feature_io

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_DIR = PROJECT_ROOT / "data_proc_3d" / "results" / "dataset" / "original"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pt", type=str, default=None,
                        help="One .pt take to load (default: first .pt in dataset/original).")
    parser.add_argument("--in-dir", type=str, default=None,
                        help="Build a one-row-per-.pt overview of this folder instead of loading one take.")
    return parser.parse_args()


def tensor_dict_to_df(tensors: dict) -> pd.DataFrame:
    """Flat {name: (T,) or (T, n) tensor/array} dict -> (T, cols) DataFrame;
    (T, n) entries become name_0 .. name_{n-1} columns."""
    columns = {}
    for name, tensor in tensors.items():
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        if array.ndim == 1:
            columns[name] = array
        else:
            array = array.reshape(len(array), -1)
            for i in range(array.shape[1]):
                columns[f"{name}_{i}"] = array[:, i]
    return pd.DataFrame(columns)


def load_take(pt_path: Path):
    """Returns (metadata, features_df, labels_df, panel_groups), both
    DataFrames indexed by source-clip frame number when their lengths match."""
    data = io_utils.load_torch(pt_path)
    metadata = data["metadata"]
    # "features" is saved as {panel_key: (T, n_cols)}; unpack to real column names
    feature_dict, panel_groups = feature_io.to_feature_dict(data["features"], metadata["panel_columns"])
    features_df = tensor_dict_to_df(feature_dict)
    labels_df = tensor_dict_to_df(data["labels"])

    if len(features_df) == len(labels_df):
        # segment_data.py chunks carry frame_start; original/augmented takes start at 0
        frame_start = metadata.get("frame_start", 0)
        index = pd.RangeIndex(frame_start, frame_start + len(features_df), name="frame")
        features_df.index = index
        labels_df.index = index
    return metadata, features_df, labels_df, panel_groups


def n_nonfinite(df: pd.DataFrame) -> pd.Series:
    """Per-column count of NaN/inf values."""
    return (~np.isfinite(df.select_dtypes(include="number"))).sum()


def describe_with_nonfinite(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = df.describe().T
    stats_df["n_nonfinite"] = n_nonfinite(df)
    return stats_df


def label_counts(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Frames per class id (rows) for each integer label column."""
    int_cols = labels_df.select_dtypes(include="integer").columns
    counts_df = labels_df[int_cols].apply(pd.Series.value_counts).fillna(0).astype(int)
    return counts_df.sort_index().rename_axis("class_id")


def build_overview(in_dir: Path) -> pd.DataFrame:
    """One row per .pt: frame counts, column counts, NaN/inf, unlabeled frames, classes present."""
    rows = []
    for pt_path in io_utils.iter_files(in_dir, extension=".pt"):
        try:
            metadata, features_df, labels_df, _ = load_take(pt_path)
        except Exception as exc:
            rows.append({"file": pt_path.name, "error": repr(exc)})
            continue
        row = {
            "file": pt_path.name,
            "T_features": len(features_df),
            "T_labels": len(labels_df),
            "frame_start": metadata.get("frame_start", 0),
            "n_feature_cols": features_df.shape[1],
            "n_label_cols": labels_df.shape[1],
            "nonfinite_feature_cols": int((n_nonfinite(features_df) > 0).sum()),
            "nonfinite_label_cols": int((n_nonfinite(labels_df) > 0).sum()),
            "fps": metadata.get("fps"),
        }
        if "step_id" in labels_df:
            row["unlabeled_frames"] = int((labels_df["step_id"] == -1).sum())
            row["step_ids"] = sorted(labels_df["step_id"].unique().tolist())
        if "status_id" in labels_df:
            row["status_ids"] = sorted(labels_df["status_id"].unique().tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if args.in_dir and not args.pt:
        in_dir = Path(args.in_dir)
        overview_df = build_overview(in_dir)
        print(f"overview_df: {len(overview_df)} .pt file(s) under {in_dir}")
    else:
        pt_path = Path(args.pt) if args.pt else next(io_utils.iter_files(DEFAULT_IN_DIR, extension=".pt"))
        metadata, features_df, labels_df, panel_groups = load_take(pt_path)

        take_df = pd.concat([features_df, labels_df], axis=1)
        panel_dfs = {panel: features_df[cols] for panel, cols in panel_groups.items()}

        feature_stats_df = describe_with_nonfinite(features_df)
        feature_stats_df.insert(0, "panel", pd.Series(
            {col: panel for panel, cols in panel_groups.items() for col in cols}))
        label_stats_df = describe_with_nonfinite(labels_df.select_dtypes(include="floating"))
        label_counts_df = label_counts(labels_df)

        listed_columns = [col for cols in metadata["panel_columns"].values() for col in cols]
        missing_panel_columns = [col for col in listed_columns if col not in features_df.columns]

        print(f"{pt_path.name}: features_df {features_df.shape}, labels_df {labels_df.shape}")
        if len(features_df) != len(labels_df):
            print(f"!! frame count mismatch: features T={len(features_df)}, labels T={len(labels_df)}")
        if missing_panel_columns:
            print(f"!! {len(missing_panel_columns)} panel_columns entries missing from features_df")
        for name, stats_df in (("feature", feature_stats_df), ("label", label_stats_df)):
            bad = stats_df.index[stats_df["n_nonfinite"] > 0].tolist()
            if bad:
                print(f"!! {len(bad)} {name} column(s) with NaN/inf: {bad[:10]}")

    # Pause here with everything above in main()'s Locals -- only under the
    # VS Code debugger, since a terminal run would otherwise drop into pdb.
    # debugpy stops on the line AFTER the breakpoint() call, hence the print.
    if "debugpy" in sys.modules:
        breakpoint()
        print("data_analysis: resumed, exiting")


if __name__ == "__main__":
    main()
