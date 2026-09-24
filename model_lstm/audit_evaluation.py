"""Re-evaluate one checkpoint and audit its raw/normalized data without training.

Outputs go to RUN/evaluation_audit; existing experiment artifacts are untouched.
The notebook's saved loading messages preserve the historical file order.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from LSTM_databuilder import AssistSequenceDataset
from LSTM_model_train import AssistLSTM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_proc_2d.app.build_norm_dataset import split_by_uid


def file_hash(path):
    # Python 3.10 does not provide hashlib.file_digest.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def historical_paths(notebook, split, directory):
    prefix = {"train": "Loading dataset from ",
              "val": "Loading validation dataset from ",
              "test": "Loading test dataset from "}[split]
    paths = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            for line in "".join(output.get("text", [])).splitlines():
                if line.startswith(prefix):
                    paths.append(directory / Path(line[len(prefix):].removesuffix("...")).name)
    current = set(directory.glob("*.npz"))
    if len(paths) != len(set(paths)) or set(paths) != current:
        raise ValueError(f"{split}: saved notebook file list differs from current data")
    return paths


def infer(model, dataset, config, device):
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        drop_last=False, num_workers=0)
    outputs = []
    with torch.inference_mode():
        for x, y in loader:
            logits, progress, mistake = model(x.to(device))
            prob = logits.softmax(dim=1)
            outputs.append(np.column_stack([
                logits.argmax(1).cpu().numpy(), progress.cpu().numpy(),
                mistake.argmax(1).cpu().numpy(),
                torch.nn.functional.cross_entropy(logits, y["step"].to(device),
                                                  reduction="none").cpu().numpy(),
                torch.nn.functional.cross_entropy(mistake, y["mistake"].to(device),
                                                  reduction="none").cpu().numpy(),
                prob[:, [0, 3, 5, 6]].cpu().numpy(),
            ]))
    return np.concatenate(outputs)


def window_metadata(dataset, path, split):
    rows = []
    uid = re.search(r"uid-(\d+)", path.name).group(1)
    augmentation = re.search(r"aug-(\d+)", path.name).group(1)
    for start in dataset.indices:
        end = start + dataset.window_size
        target = end - 1 + dataset.predict_offset
        labels = dataset.step[start:end]
        counts = np.bincount(labels, minlength=8)
        changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        rows.append({
            "split": split, "file": path.name, "participant": uid,
            "augmentation": augmentation, "start_frame": start,
            "target_frame": target, "true_step": int(dataset.step[target]),
            "true_progress": float(dataset.progress[target]),
            "true_mistake": int(dataset.mistake[target]),
            "pure": bool(np.all(labels == dataset.step[target])),
            "majority_step": int(counts.argmax()),
            "target_fraction": float(counts[dataset.step[target]] / len(labels)),
            "step5_fraction": float(counts[5] / len(labels)),
            "step6_fraction": float(counts[6] / len(labels)),
            "frames_since_transition": int(len(labels) - changes[-1]) if len(changes) else len(labels),
            "transitions": "|".join(f"{labels[i-1]}>{labels[i]}" for i in changes),
        })
    return pd.DataFrame(rows)


def class_metrics(frame, groups):
    rows = []
    for group, subset in frame.groupby(groups, sort=False):
        group = group if isinstance(group, tuple) else (group,)
        for label in range(8):
            truth = subset.true_step == label
            pred = subset.pred_step == label
            support, predicted = int(truth.sum()), int(pred.sum())
            correct = int((truth & pred).sum())
            rows.append(dict(zip(groups, group), step=label, support=support,
                             predicted=predicted, correct=correct,
                             recall=correct / support if support else np.nan,
                             prediction_rate=predicted / len(subset)))
    return pd.DataFrame(rows)


def save_summaries(frame, out):
    class_metrics(frame, ["split"]).to_csv(out / "class_metrics.csv", index=False)
    class_metrics(frame, ["split", "participant"]).to_csv(out / "participant_recall.csv", index=False)
    class_metrics(frame, ["split", "augmentation"]).to_csv(out / "augmentation_recall.csv", index=False)
    rows = []
    for (split, uid, label, pure), group in frame[frame.true_step.isin([0, 3, 5, 6])].groupby(
            ["split", "participant", "true_step", "pure"]):
        rows.append(dict(split=split, participant=uid, step=int(label), pure=bool(pure),
                         support=len(group), correct=int((group.pred_step == label).sum()),
                         predicted_5=int((group.pred_step == 5).sum()),
                         predicted_6=int((group.pred_step == 6).sum()),
                         majority_disagrees=int((group.majority_step != label).sum())))
    groups = pd.DataFrame(rows)
    groups.to_csv(out / "window_groups_by_participant.csv", index=False)
    aggregate = groups.groupby(["split", "step", "pure"], as_index=False).sum(numeric_only=True)
    aggregate["recall"] = aggregate.correct / aggregate.support
    aggregate["predicted_6_rate"] = aggregate.predicted_6 / aggregate.support
    aggregate.to_csv(out / "window_groups.csv", index=False)
    for split, group in frame.groupby("split", sort=False):
        matrix = pd.crosstab(group.true_step, group.pred_step).reindex(
            index=range(8), columns=range(8), fill_value=0)
        matrix.to_csv(out / f"{split}_confusion.csv")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=Path(__file__).with_name("LSTM_train_mirror.ipynb"))
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    if config["mode"] != "seq2one" or config["predict_offset"] != 0:
        raise ValueError("This audit analyzes last-frame, zero-offset seq2one labels")
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    out = args.run / "evaluation_audit"
    out.mkdir(exist_ok=True)
    stats = {}
    for path in sorted(args.norm.parent.glob("norm*.npz")):
        with np.load(path, allow_pickle=False) as data:
            stats[path.name] = {key: data[key] for key in data.files}
    norm = stats[args.norm.name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    model = AssistLSTM(**{key: config[key] for key in (
        "input_dim", "hidden_dim", "num_steps", "dropout", "num_layers", "num_mistakes")}).to(device)
    model.load_state_dict(torch.load(args.run / "best_model.pth", map_location=device, weights_only=True))
    model.eval()
    print(f"Checkpoint: {args.run.name}; device: {device}", flush=True)
    file_rows, feature_rows, frames, corrected = [], [], [], []
    columns_reference = None
    fit_sum, fit_square, fit_n = {}, {}, 0
    fit_features = {key: [] for key in config["feature_keys"]}
    for split in ("train", "val", "test"):
        directory = args.data_root / f"aug_{split}" / "norm"
        paths = historical_paths(notebook, split, directory)
        raw_paths = set((directory.parent / "raw").glob("*.pt"))
        expected_raw = {directory.parent / "raw" / (p.stem.removesuffix("_norm") + ".pt") for p in paths}
        if raw_paths != expected_raw:
            raise ValueError(f"{split}: raw/normalized file sets differ")
        split_frames = []
        for index, path in enumerate(paths):
            raw_path = directory.parent / "raw" / (path.stem.removesuffix("_norm") + ".pt")
            raw = torch.load(raw_path, map_location="cpu", weights_only=False)
            dataset = AssistSequenceDataset(path, **{k: config[k] for k in (
                "window_size", "stride", "predict_offset", "mode", "feature_keys")})
            if sum(x.shape[1] for x in dataset.features) != config["input_dim"]:
                raise ValueError(f"{path}: input dimensions differ")
            columns = {key: raw["metadata"]["panel_columns"][key] for key in config["feature_keys"]}
            if columns_reference is None:
                columns_reference = columns
            if columns != columns_reference:
                raise ValueError(f"{path}: feature column order differs")
            for key, actual in (("task_id", dataset.step), ("mistake", dataset.mistake),
                                ("task_progress", dataset.progress)):
                expected = raw["labels"][key].numpy()
                if key == "task_progress":
                    expected = expected.astype(np.float32) / 100.0
                if not np.array_equal(actual, expected):
                    raise ValueError(f"{path}: {key} differs from raw labels")
            if dataset.step.min() < 0 or dataset.step.max() >= config["num_steps"]:
                raise ValueError(f"{path}: labels outside configured mapping")
            intended_features = []
            matches = {name: True for name in stats}
            max_error = 0.0
            content_hash = hashlib.sha256()
            for key, actual in zip(config["feature_keys"], dataset.features):
                content_hash.update(actual.tobytes())
                raw_feature = raw["features"][key].numpy()
                if not np.isfinite(actual).all() or not np.isfinite(raw_feature).all():
                    raise ValueError(f"{path}: nonfinite features")
                for name, candidate in stats.items():
                    std = np.where(candidate[f"{key}_std"] < 1e-6, 1.0, candidate[f"{key}_std"])
                    expected = ((raw_feature - candidate[f"{key}_mean"]) / std).astype(np.float32)
                    exact = np.array_equal(actual, expected)
                    error = float(np.max(np.abs(actual - expected)))
                    matches[name] &= exact
                    feature_rows.append(dict(split=split, file=path.name, feature=key,
                                             norm=name, exact=exact, max_abs_error=error))
                    if name == args.norm.name:
                        intended_features.append(expected)
                        max_error = max(max_error, error)
                if split == "train":
                    fit_features[key].append((path.name, raw_feature))
                    values = raw_feature.astype(np.float64)
                    fit_sum[key] = fit_sum.get(key, 0) + values.sum(axis=0)
                    fit_square[key] = fit_square.get(key, 0) + np.square(values).sum(axis=0)
            if split == "train":
                fit_n += dataset.T
            for label in (dataset.step, dataset.progress, dataset.mistake):
                content_hash.update(label.tobytes())
            # Verify the exact local NPY cache version addressed by the notebook.
            stat = path.stat()
            keys = config["feature_keys"] + ["task_id", "task_progress", "mistake"]
            signature = json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns, keys])
            cache = Path(config["dataset_cache_dir"]) / hashlib.sha256(signature.encode()).hexdigest()
            cache_matches = []
            for i, actual in enumerate(dataset.features + [dataset.step, dataset.progress, dataset.mistake]):
                cached = cache / f"{i}.npy"
                cache_matches.append(np.array_equal(np.load(cached, mmap_mode="r"), actual) if cached.exists() else None)
            result = window_metadata(dataset, path, split)
            prediction_columns = ["pred_step", "pred_progress", "pred_mistake", "step_loss", "mistake_loss",
                                  "prob_0", "prob_3", "prob_5", "prob_6"]
            result[prediction_columns] = infer(model, dataset, config, device)
            result[["pred_step", "pred_mistake"]] = result[["pred_step", "pred_mistake"]].astype(int)
            split_frames.append(result)
            if not matches[args.norm.name]:
                dataset.features = intended_features
                alternate = result.copy()
                alternate[prediction_columns] = infer(model, dataset, config, device)
                corrected.append(alternate)
            file_rows.append(dict(split=split, file=path.name, frames=dataset.T, windows=len(dataset),
                                  participant=result.participant.iloc[0],
                                  matched_norms="|".join(name for name, match in matches.items() if match),
                                  intended_norm_exact=matches[args.norm.name], norm_max_error=max_error,
                                  cache_arrays_present=sum(x is not None for x in cache_matches),
                                  cache_arrays_differ=sum(x is False for x in cache_matches),
                                  raw_mtime_ns=raw_path.stat().st_mtime_ns,
                                  norm_mtime_ns=stat.st_mtime_ns, norm_size=stat.st_size,
                                  selected_features_and_labels_sha256=content_hash.hexdigest()))
            if (index + 1) % 10 == 0 or index + 1 == len(paths):
                print(f"{split}: {index + 1}/{len(paths)} files; matching intended stats: "
                      f"{sum(r['intended_norm_exact'] for r in file_rows if r['split'] == split)}", flush=True)
        combined = pd.concat(split_frames, ignore_index=True)
        combined.to_csv(out / f"{split}_predictions.csv", index=False)
        frames.append(combined)
    frame = pd.concat(frames, ignore_index=True)
    files = pd.DataFrame(file_rows)
    files.to_csv(out / "file_audit.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out / "normalization_comparison.csv", index=False)
    if corrected:
        pd.concat(corrected, ignore_index=True).to_csv(out / "renormalized_predictions.csv", index=False)
    save_summaries(frame, out)
    metrics = {}
    for split, group in frame.groupby("split", sort=False):
        metrics[split] = dict(windows=len(group), accuracy=float((group.true_step == group.pred_step).mean()),
                              step_loss=float(group.step_loss.mean()),
                              progress_loss=float(((group.true_progress - group.pred_progress) ** 2).mean()),
                              mistake_loss=float(group.mistake_loss.mean()))
    historical = pd.read_csv(args.run / "test_predictions.csv")
    test = frame[frame.split == "test"].reset_index(drop=True)
    comparisons = {key: bool(np.array_equal(test[key].to_numpy(), historical[key].to_numpy()))
                   for key in ["true_step", "pred_step", "true_mistake", "pred_mistake"]}
    comparisons["max_progress_abs_difference"] = float(np.max(np.abs(test.pred_progress - historical.pred_progress)))
    summary = json.loads((args.run / "summary.json").read_text())
    participants = {split: sorted(group.participant.unique().tolist()) for split, group in frame.groupby("split")}
    overlaps = {f"{a}/{b}": sorted(set(participants[a]) & set(participants[b]))
                for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    reconstructed = split_by_uid(sorted(name.removesuffix("_norm.npz") + ".pt" for name in files.file))
    split_matches = all(
        set(names) == {name.removesuffix("_norm.npz") + ".pt" for name in files.loc[files.split == split, "file"]}
        for split, names in zip(("train", "val", "test"), reconstructed)
    )
    fit_order = {name.removesuffix(".pt") + "_norm.npz": i for i, name in enumerate(reconstructed[0])}
    fit = {}
    for key in config["feature_keys"]:
        mean = fit_sum[key] / fit_n
        std = np.sqrt(np.maximum(0, fit_square[key] / fit_n - mean ** 2)) + 1e-6
        fit[key] = dict(mean_max_abs_difference=float(np.max(np.abs(mean - norm[f"{key}_mean"]))),
                        std_max_abs_difference=float(np.max(np.abs(std - norm[f"{key}_std"]))),
                        mean_error_in_std_units=float(np.max(np.abs(mean - norm[f"{key}_mean"]) / std)),
                        std_max_relative_difference=float(np.max(np.abs(std - norm[f"{key}_std"]) / std)))
        # Reproduce compute_global_norm_stats: sorted files, concatenation,
        # NumPy's default float32 accumulation, then +1e-6 in float32.
        source_arrays = fit_features.pop(key)
        values = np.concatenate([array for _, array in sorted(source_arrays)], axis=0)
        mean32 = values.mean(axis=0).astype(np.float32)
        std32 = (values.std(axis=0) + 1e-6).astype(np.float32)
        fit[key]["sorted_float32_mean_exact"] = bool(np.array_equal(mean32, norm[f"{key}_mean"]))
        fit[key]["sorted_float32_std_exact"] = bool(np.array_equal(std32, norm[f"{key}_std"]))
        del values
        if split_matches:
            # On the first build, split_by_uid returns participant-group order,
            # rather than the sorted order used when existing splits are reused.
            values = np.concatenate([a for _, a in sorted(source_arrays, key=lambda item: fit_order[item[0]])])
            mean32 = values.mean(axis=0).astype(np.float32)
            std32 = (values.std(axis=0) + 1e-6).astype(np.float32)
            fit[key]["initial_split_float32_mean_exact"] = bool(np.array_equal(mean32, norm[f"{key}_mean"]))
            fit[key]["initial_split_float32_std_exact"] = bool(np.array_equal(std32, norm[f"{key}_std"]))
            del values
        del source_arrays
    train_counts = np.bincount(frame.loc[frame.split == "train", "true_step"], minlength=8)
    audit = dict(run=str(args.run.resolve()), checkpoint_sha256=file_hash(args.run / "best_model.pth"),
                 config_sha256=file_hash(args.run / "config.json"), notebook_sha256=file_hash(args.notebook),
                 normalization_sha256={p.name: file_hash(p) for p in args.norm.parent.glob("norm*.npz")},
                 feature_keys=config["feature_keys"], feature_columns=columns_reference,
                 normalization_all_files_exact=bool(files.intended_norm_exact.all()),
                 cache_arrays_differ=int(files.cache_arrays_differ.sum()),
                 train_counts_match_config=bool(np.array_equal(train_counts, config["step_class_counts"])),
                 train_counts=train_counts.tolist(), evaluation="sequential, no resampling, drop_last=False, model.eval",
                 metrics=metrics, historical_test_comparison=comparisons,
                 val_accuracy_difference=metrics["val"]["accuracy"] - summary["best_val_acc"],
                 val_loss_difference=sum(metrics["val"][k] * config[f"lambda_{k.removesuffix('_loss')}"]
                                         for k in ["step_loss", "progress_loss", "mistake_loss"]) - summary["best_val_loss"],
                 participants=participants, participant_overlap=overlaps,
                 reconstructed_split_matches=split_matches,
                 normalization_fit_float64_check=fit, train_frames=fit_n,
                 limitation="Historical run has no immutable data/norm manifest; this audits current files and reproduces saved metrics.")
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
