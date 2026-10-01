"""Shared A/B/C LOSO entry point. Reads existing claude_l1 exports; never rebuilds them.

Full experiment: python -B -m model_lstm.ablation_step_supervision.run_loso
Smoke run:      python -B -m model_lstm.ablation_step_supervision.run_loso --folds 1 --epochs 1
Direct execution from an IDE works too. Importing this file does not start training.
"""

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_lstm.ablation_step_supervision.data import FEATURE_KEYS, CLASS_NAMES, load_fold
from model_lstm.ablation_step_supervision.engine import fit, run_epoch
from model_lstm.ablation_step_supervision.models import MODELS
from model_lstm.ablation_step_supervision.report import save_confusion, summarize


DEFAULT_DATA = Path(
    "G:/.shortcut-targets-by-id/1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC/"
    "ITECH_Thesis/Videos/dataset/skeleton_3d/ceiling_panel_installation_04/claude_l1"
)
SETTINGS = dict(
    input_dim=91, hidden_dim=64, num_layers=1, dropout=0.2,
    samples=40, sample_interval=3, raw_fps=30, input_fps=10, window_hop=10, predict_offset=0,
    batch_size=32, optimizer="Adam", learning_rate=1e-4, weight_decay=0.01,
    scheduler=None, gradient_clip=None, sampler="RandomSampler", sampler_replacement=False,
    class_weights=None, pos_weight=None, class7_factor=1.0,
    progress_head="scalar_linear", mistake_head="two_class_CE",
    loss_coefficients=dict(step=1.0, progress=1.0, mistake=1.0, background=0.2),
    background_threshold=0.5, target="task_id", background_target="task_id == 7",
    selection_metric="val_macro_f1_8", labels=list(range(8)),
    feature_keys=list(FEATURE_KEYS), class_names=list(CLASS_NAMES),
    step_supervision=dict(A="8-class CE, all windows", B="7-class CE, task_id != 7",
                          C="7-lane soft-peak BCE, task_id != 7"),
)


def run_loso(data_root, run_root, arms=("A", "B", "C"), folds=tuple(range(1, 16)),
             epochs=12, seed=42, device=None, cache_dir=None):
    if epochs <= 0 or not arms or not folds or len(set(arms)) != len(arms) or len(set(folds)) != len(folds):
        raise ValueError("Choose positive epochs and unique nonempty arms/folds")
    if set(arms) - set(MODELS) or set(folds) - set(range(1, 16)):
        raise ValueError("Arms must be A/B/C; folds must be 1..15")
    data_root, run_root = Path(data_root).resolve(), Path(run_root).resolve()
    for uid in folds:
        for name in ("dataset_manifest.json", "norm_stats.npz"):
            if not (data_root / f"fold_{uid:02d}" / name).is_file():
                raise FileNotFoundError(data_root / f"fold_{uid:02d}" / name)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache_dir = Path(cache_dir) if cache_dir else ROOT / "model_lstm/.dataset_cache/ablation_step_supervision"
    protocol = dict(SETTINGS, arms=list(arms), folds=list(folds), epochs=epochs, seed=seed,
                    data_root=str(data_root), device=str(device),
                    experiment="step_supervision_ABC_v1", standard_run=(epochs == 12 and len(folds) == 15))
    resuming = run_root.exists()
    if resuming:
        previous = json.loads((run_root / "protocol.json").read_text(encoding="utf-8"))
        differences = [key for key in protocol if key not in ("arms", "folds", "device", "standard_run")
                       and previous.get(key) != protocol[key]]
        if differences:
            raise ValueError(f"Existing experiment settings differ: {', '.join(differences)}")
        # Validate every existing fold before changing anything, including unselected arms.
        for arm in MODELS:
            for config_path in (run_root / arm).glob("fold_*/config.json"):
                directory = config_path.parent
                source = data_root / directory.name
                config = json.loads(config_path.read_text(encoding="utf-8"))
                for norm_path in (source / "norm_stats.npz", directory / "norm_stats.npz"):
                    if norm_path.exists() and hashlib.sha256(norm_path.read_bytes()).hexdigest() != config["normalization_sha256"]:
                        raise ValueError(f"Existing normalization differs: {norm_path}")
                saved_manifest = directory / "dataset_manifest.json"
                if saved_manifest.exists() and json.loads(saved_manifest.read_text()) != json.loads((source / "dataset_manifest.json").read_text()):
                    raise ValueError(f"Existing dataset manifest differs: {source}")
        protocol["arms"] = sorted(set(previous["arms"]) | set(arms))
        protocol["folds"] = sorted(set(previous["folds"]) | set(folds))
        protocol["standard_run"] = epochs == 12 and len(protocol["folds"]) == 15
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    # Preserve the actual experiment implementation alongside its results.
    source_dir = run_root / "source"
    if not resuming:
        source_dir.mkdir()
        for name in ("models.py", "data.py", "engine.py", "report.py", "run_loso.py", "README.md"):
            shutil.copy2(Path(__file__).with_name(name), source_dir / name)
    summarize(run_root, MODELS)
    for uid in folds:
        pending = []
        for arm in arms:
            if (run_root / arm / f"fold_{uid:02d}" / "test_results.json").is_file():
                print(f"Skipping completed arm {arm}, test UID {uid:02d}", flush=True)
            else:
                pending.append(arm)
        if not pending:
            continue
        print(f"\nLoading shared fold {uid:02d} on {device}", flush=True)
        datasets, manifest = load_fold(data_root, uid, cache_dir / f"fold_{uid:02d}",
                                       SETTINGS["samples"], SETTINGS["sample_interval"], SETTINGS["window_hop"])
        window_counts = {split: len(dataset) for split, dataset in datasets.items()}
        print(f"Windows: {window_counts}", flush=True)
        fold_seed = seed + uid - 1
        for arm in pending:
            print(f"\nArm {arm}, test UID {uid:02d}", flush=True)
            run_dir = run_root / arm / f"fold_{uid:02d}"
            if run_dir.exists():
                print("  Incomplete run: restarting from epoch 1", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            # fit appends metrics; a restarted fold must not retain its old rows.
            (run_dir / "metrics.csv").unlink(missing_ok=True)
            config = dict(protocol, arm=arm, test_uid=uid, fold_seed=fold_seed,
                          subjects=manifest["subjects"], window_counts=window_counts,
                          normalization_id=manifest["normalization_id"],
                          normalization_sha256=manifest["normalization_sha256"],
                          normalization_file="norm_stats.npz", test_checkpoint="best_macro_f1.pth")
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            shutil.copy2(data_root / f"fold_{uid:02d}" / "norm_stats.npz", run_dir / "norm_stats.npz")
            shutil.copy2(data_root / f"fold_{uid:02d}" / "dataset_manifest.json", run_dir / "dataset_manifest.json")
            loaders = {split: DataLoader(
                dataset, batch_size=SETTINGS["batch_size"], shuffle=split == "train",
                generator=torch.Generator().manual_seed(fold_seed), num_workers=0,
                pin_memory=device.type == "cuda", drop_last=False,
            ) for split, dataset in datasets.items()}
            torch.manual_seed(fold_seed)
            model = MODELS[arm](**{key: SETTINGS[key] for key in ("input_dim", "hidden_dim", "dropout")}).to(device)
            # Keep dropout randomness independent of head construction.
            torch.manual_seed(fold_seed + 100_000)
            fit(model, loaders["train"], loaders["val"], device, run_dir, epochs,
                SETTINGS["learning_rate"], SETTINGS["weight_decay"])
            results, cm, predictions = run_epoch(model, loaders["test"], device, collect=True)
            parts = datasets["test"].datasets
            predictions.insert(0, "test_uid", uid)
            predictions["file"] = np.concatenate([np.repeat(part.path.name, len(part)) for part in parts])
            predictions["target_frame"] = np.concatenate([part.target_frames for part in parts])
            predictions.to_csv(run_dir / "test_predictions.csv", index=False)
            save_confusion(cm, run_dir, f"{arm}: test UID {uid:02d}\nAll 8 classes, including background")
            best = json.loads((run_dir / "best_macro_f1.json").read_text())
            results.update(test_uid=uid, best_epoch=best["epoch"])
            # Write last: summary only includes folds whose prediction/report export completed.
            marker = run_dir / "test_results.json.tmp"
            marker.write_text(json.dumps(results, indent=2), encoding="utf-8")
            marker.replace(run_dir / "test_results.json")
            print(f"  Test macro-F1(8): {results['macro_f1_8']:.4f}; best epoch {best['epoch']}", flush=True)
            del model, loaders, predictions
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            summarize(run_root, MODELS)
        del datasets
        gc.collect()
    print(f"\nResults saved: {run_root}")
    return run_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA, help="Existing claude_l1 directory with 15 fold exports")
    parser.add_argument("--output-dir", type=Path, help="New or existing experiment directory; completed arm/folds are skipped")
    parser.add_argument("--arms", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 16), default=list(range(1, 16)))
    parser.add_argument("--epochs", type=int, default=12, help="12 for the agreed experiment; smaller values for smoke checks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(__file__).parent / "runs" / datetime.now().strftime("ABC_%Y-%m-%d_%H-%M-%S_%f")
    run_loso(args.data_root, output, args.arms, args.folds, args.epochs, args.seed, args.device, args.cache_dir)


if __name__ == "__main__":
    main()
