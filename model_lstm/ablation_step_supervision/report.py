"""Counts-first confusion plots and paired, per-participant LOSO summaries."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .data import CLASS_NAMES
from .engine import class_scores


def save_confusion(cm, directory, title):
    directory = Path(directory)
    precision, recall, f1, support = class_scores(cm)
    pd.DataFrame(dict(class_name=CLASS_NAMES, precision=precision, recall=recall,
                      f1=f1, support=support)).to_csv(directory / "per_class.csv", index=False)
    norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros((8, 8)), where=cm.sum(1, keepdims=True) != 0)
    for matrix, suffix, fmt in ((cm, "", "d"), (norm, "_norm", ".2f")):
        table = pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)
        table.index.name = "true_step"
        table.to_csv(directory / f"confusion_matrix{suffix}.csv")
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
            ax=ax, cmap="Blues", xticks_rotation=45, values_format=fmt,
            im_kw={"vmin": 0, "vmax": 1} if suffix else None,
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(directory / f"confusion_matrix{suffix}.png", dpi=160)
        plt.close(fig)


def summarize(run_root, arms):
    comparisons, per_arm = [], {}
    for arm in arms:
        directory = Path(run_root) / arm
        rows, recalls, pooled = [], [], np.zeros((8, 8), dtype=np.int64)
        for result_path in sorted(directory.glob("fold_*/test_results.json")):
            result = json.loads(result_path.read_text())
            cm = pd.read_csv(result_path.parent / "confusion_matrix.csv", index_col=0).to_numpy(dtype=np.int64)
            pooled += cm
            rows.append(result)
            recalls.append(dict(test_uid=result["test_uid"], **dict(zip(CLASS_NAMES, class_scores(cm)[1]))))
        if not rows:
            continue
        frame = pd.DataFrame(rows).sort_values("test_uid")
        frame.to_csv(directory / "fold_results.csv", index=False)
        pd.DataFrame(recalls).to_csv(directory / "participant_recall.csv", index=False)
        metrics = [k for k in frame.columns if k not in ("test_uid", "best_epoch")]
        summary = dict(completed_folds=len(frame), test_uids=frame.test_uid.tolist(), std_ddof=0,
                       primary_metric="macro_f1_8", metrics={
                           k: dict(mean=float(frame[k].mean()), std=float(frame[k].std(ddof=0))) for k in metrics})
        (directory / "loso_summary.json").write_text(json.dumps(summary, indent=2))
        save_confusion(pooled, directory, f"{arm}: pooled test predictions, {len(frame)} folds\nAll 8 classes, including background")
        comparisons.append(dict(arm=arm, folds=len(frame), macro_f1_8_mean=frame.macro_f1_8.mean(),
                                macro_f1_8_std=frame.macro_f1_8.std(ddof=0),
                                accuracy_8_mean=frame.accuracy_8.mean(), background_f1_mean=frame.background_f1.mean()))
        per_arm[arm] = frame.set_index("test_uid")
    if comparisons:
        pd.DataFrame(comparisons).to_csv(Path(run_root) / "comparison.csv", index=False)
    pairs = []
    for first, second in (("A", "B"), ("B", "C"), ("A", "C")):
        if first in per_arm and second in per_arm:
            delta = (per_arm[second].macro_f1_8 - per_arm[first].macro_f1_8).dropna()
            for uid, value in delta.items():
                pairs.append(dict(comparison=f"{first}_to_{second}", test_uid=uid, macro_f1_8_delta=value))
    if pairs:
        pd.DataFrame(pairs).to_csv(Path(run_root) / "paired_differences.csv", index=False)
