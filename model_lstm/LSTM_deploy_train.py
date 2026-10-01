"""Train the final L1 LSTM on all 15 participants; no validation/test selection.

Run from the repository root: python -B -m model_lstm.LSTM_deploy_train
Direct execution / IDE Run also works. Use --help for paths and epoch overrides.
Defaults select the median best validation epoch from the completed L1 LOSO run.
Importing this module does not export data or start training.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from data_proc_2d.app.build_norm_dataset_tune import (
    FEATURE_KEYS, L1_PROFILE, NormDatasetBuilder, compute_global_norm_stats,
)
from model_lstm.LSTM_databuilder_tune import AssistSequenceDataset, TASK_NAMES, training_pos_weights
from model_lstm.LSTM_engine_tune import L1_SETTINGS, run_epoch
from model_lstm.LSTM_model_train_tune import AssistLSTM, AssistLoss


DEFAULT_DATA_ROOT = Path(
    "G:/.shortcut-targets-by-id/1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC/"
    "ITECH_Thesis/Videos/dataset/skeleton_3d/ceiling_panel_installation_04"
)
DEFAULT_LOSO_RUN = ROOT / "model_lstm/runs_3d_tune/L1_LOSO_2026-09-29_11-19-36_618304"


def select_epochs(loso_run_dir, epochs=None):
    """Use validation-selected epochs only; never read test scores."""
    if epochs is not None:
        if not 1 <= epochs <= L1_SETTINGS["epochs"]:
            raise ValueError("Epoch override must be within the L1 schedule (1..12)")
        return epochs, {"method": "explicit_override", "epochs": epochs}
    chosen = []
    for uid in range(1, 16):
        fold = Path(loso_run_dir) / f"fold_{uid:02d}"
        config = json.loads((fold / "config.json").read_text(encoding="utf-8"))
        if any(config.get(key) != value for key, value in L1_SETTINGS.items()):
            raise ValueError(f"LOSO settings differ from current L1: {fold}")
        summary = json.loads((fold / "summary_step_macro_f1.json").read_text(encoding="utf-8"))
        epoch = summary["best_epoch"]
        if not isinstance(epoch, int) or not 1 <= epoch <= L1_SETTINGS["epochs"]:
            raise ValueError(f"Invalid best validation epoch: {fold}")
        chosen.append(epoch)
    return int(median(chosen)), {
        "method": "median_of_15_best_validation_task_f1_epochs",
        "loso_run_dir": str(Path(loso_run_dir).resolve()), "fold_epochs": chosen,
    }


def full_training_files(data_root):
    """Every original plus exactly its aug-01 mirror, including former val/test UIDs."""
    data_root = Path(data_root)
    originals = sorted((data_root / "original").glob("*.pt"))
    subjects = {int(re.search(r"uid-(\d+)", p.name)[1]) for p in originals}
    if subjects != set(range(1, 16)):
        raise ValueError(f"Expected originals for UID 1..15, found {sorted(subjects)}")
    mirrors = {}
    for directory in [data_root / "augmented_mirror"] + [
        data_root / split / "raw" for split in ("aug_train", "aug_val", "aug_test")
    ]:
        for path in sorted(directory.rglob("*_aug-01.pt")):
            key = path.stem.removesuffix("_aug-01")
            if key in mirrors and mirrors[key].resolve() != path.resolve():
                raise ValueError(f"Ambiguous mirror copies: {key}")
            mirrors[key] = path
    files = []
    for original in originals:
        if original.stem not in mirrors:
            raise ValueError(f"Missing aug-01 mirror: {original.name}")
        files.extend([original, mirrors[original.stem]])
    return files


def prepare_full_dataset(data_root, export_dir):
    """Fit new full-training norm and export NPZs, reusing only matching exports."""
    paths = full_training_files(data_root)
    export_dir = Path(export_dir)
    identity = dict(
        profile=L1_PROFILE, feature_keys=FEATURE_KEYS, subjects=list(range(1, 16)),
        sources=[dict(path=str(p.resolve()), size=p.stat().st_size,
                      mtime_ns=p.stat().st_mtime_ns) for p in paths],
        statistics_frame_step=7, statistics_dtype="float64", std_ddof=0,
    )
    manifest_path = export_dir / "dataset_manifest.json"
    stats_path = export_dir / "norm_stats.npz"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Full-data sources/protocol changed; choose a new --export-dir")
        if hashlib.sha256(stats_path.read_bytes()).hexdigest() != saved["normalization_sha256"]:
            raise ValueError(f"Normalization file changed: {stats_path}")
        if not all((export_dir / name).is_file() for name in saved["files"]):
            raise ValueError("Incomplete full-data export; choose a new --export-dir")
        return saved
    # Do not overwrite a fold export or a partially completed previous export.
    if export_dir.exists() and any(export_dir.iterdir()):
        raise ValueError("Export directory is nonempty without a manifest; choose a new --export-dir")
    export_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_global_norm_stats(paths, profile=L1_PROFILE)
    np.savez(stats_path, **stats)
    builder = NormDatasetBuilder(stats, profile=L1_PROFILE)
    (export_dir / "train").mkdir()
    files = []
    for path in tqdm(paths, desc="Export full training data"):
        relative = f"train/{path.stem}_norm.npz"
        builder.build_dataset(path, export_dir / relative)
        files.append(relative)
    manifest = dict(identity, files=files, normalization_id=builder.normalization_id,
                    normalization_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_full_dataset(data_root, export_dir, output_dir, loso_run_dir=DEFAULT_LOSO_RUN,
                       epochs=None, seed=0, device=None):
    epochs, selection = select_epochs(loso_run_dir, epochs)
    #write in longer epoches
    epochs = 30
    output_dir, export_dir = Path(output_dir), Path(export_dir)
    if output_dir.exists():
        raise ValueError(f"Choose a new output directory: {output_dir}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Full-data L1: UID 1..15, {epochs} epochs, device={device}, seed={seed}", flush=True)
    manifest = prepare_full_dataset(data_root, export_dir)
    datasets = [AssistSequenceDataset(
        export_dir / name, window_size=L1_SETTINGS["window_size"], stride=L1_SETTINGS["stride"],
        feature_keys=FEATURE_KEYS,
        cache_dir=ROOT / "model_lstm/.dataset_cache/claude_l1_full",
        expected_normalization_id=manifest["normalization_id"],
    ) for name in manifest["files"]]
    joined = ConcatDataset(datasets)
    pos_stats = training_pos_weights(datasets, cap=L1_SETTINGS["pos_weight_cap"])
    if joined[0][0].shape[-1] != L1_SETTINGS["input_dim"]:
        raise ValueError("Feature dimensions do not match L1")
    loader = DataLoader(joined, batch_size=L1_SETTINGS["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        pin_memory=(device.type == "cuda"), drop_last=False)
    torch.manual_seed(seed)
    model = AssistLSTM(**{k: L1_SETTINGS[k] for k in
                         ("input_dim", "hidden_dim", "num_steps", "dropout", "num_layers")}).to(device)
    criterion = AssistLoss(
        pos_stats["task_pos_weight"], pos_stats["mistake_pos_weight"],
        **{f"lambda_{k}": v for k, v in L1_SETTINGS["loss_weights"].items()},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=L1_SETTINGS["learning_rate"],
                                 weight_decay=L1_SETTINGS["weight_decay"])
    # Replay the original 12-epoch LR schedule through the selected stopping epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=L1_SETTINGS["epochs"])
    config = dict(
        L1_SETTINGS, epochs=epochs, scheduler_t_max=L1_SETTINGS["epochs"], seed=seed,
        stage="full_dataset_deployment", epoch_selection=selection,
        train_uids=list(range(1, 16)), val_uids=[], test_uids=[],
        feature_keys=FEATURE_KEYS, feature_profile=L1_PROFILE, task_names=list(TASK_NAMES),
        normalization_id=manifest["normalization_id"],
        normalization_sha256=manifest["normalization_sha256"],
        normalization_file="norm_stats.npz", model_file="model_weights.pth",
        pos_weight_stats=pos_stats, training_sampler="RandomSampler", sampler_replacement=False,
        checkpoint_selection="final_epoch_fixed_before_training", input_fps=30,
        feature_transforms={"polar_azimuth": "degrees -> all sin columns, then all cos columns",
                            "ratios": "clip to [0, 4] before normalization"},
        decoding={"task": "argmax of 7 task logits", "idle": "idle logit > 0 -> class 7",
                  "mistake": "mistake logit > 0", "progress": "7 raw linear outputs, multiply by 100 for percent"},
    )
    with np.load(export_dir / "norm_stats.npz") as stats:
        start, layout = 0, []
        for key in FEATURE_KEYS:
            stop = start + len(stats[f"{key}_mean"])
            layout.append(dict(feature=key, start=start, stop=stop))
            start = stop
    config["feature_layout"] = layout
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(export_dir / "norm_stats.npz", output_dir / "norm_stats.npz")
    shutil.copy2(export_dir / "dataset_manifest.json", output_dir / "dataset_manifest.json")
    shutil.copy2(Path(__file__).with_name("LSTM_model_train_tune.py"), output_dir / "model_definition.py")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        metrics, _ = run_epoch(model, loader, criterion, device, optimizer=optimizer,
                               gradient_clip=L1_SETTINGS["gradient_clip"])
        row = dict(epoch=epoch, learning_rate=lr, **{f"train_{k}": v for k, v in metrics.items()})
        with (output_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} | train loss {metrics['loss']:.4f} "
              f"| train task F1 {metrics['step_macro_f1']:.4f}", flush=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output_dir / "model_weights.pth")
    summary = dict(completed_epochs=epochs, training_windows=len(joined), source_files=len(datasets),
                   epoch_selection=selection, checkpoint="model_weights.pth",
                   evaluation="No held-out evaluation; report LOSO results for generalization.")
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Final L1 deployment model\n\n"
        f"Trained from scratch on all 15 subjects, originals + one mirror, for {epochs} epochs.\n"
        "Load model_weights.pth with AssistLSTM from model_definition.py and the architecture in config.json.\n"
        "Apply the feature transforms in config.json, then normalize each panel with norm_stats.npz; "
        "concatenate in feature_keys order, preserving the source PT column order within each panel.\n"
        "Use 120 consecutive frames at 30 fps; outputs describe the last frame. Window hop 10 is the training stride.\n"
        "The forward result is (step_logits, progress, mistake_logit, idle_logit). "
        "step_logits already contains all 7 raw task scores, with shape [batch, 7], "
        "in config.json task_names order. No argmax is applied inside the model. "
        "Use sigmoid (not softmax) for this soft-peak BCE model; these independent "
        "response scores need not sum to one and are not calibrated confidence probabilities. "
        "Keep the idle score separate for post-processing. Class 7 is idle.\n\n"
        "## Load and return scores for post-processing\n\n"
        "Run this example from the deployment directory. The input x must already use "
        "this model's transforms, norm and feature order described above.\n\n"
        "```python\n"
        "import json\n"
        "import torch\n"
        "from model_definition import AssistLSTM\n\n"
        "with open('config.json', encoding='utf-8') as stream:\n"
        "    config = json.load(stream)\n"
        "model = AssistLSTM(**{k: config[k] for k in\n"
        "    ('input_dim', 'hidden_dim', 'num_steps', 'dropout', 'num_layers')})\n"
        "model.load_state_dict(torch.load('model_weights.pth', map_location='cpu', weights_only=True))\n"
        "model.eval()\n\n"
        "@torch.inference_mode()\n"
        "def predict(x):\n"
        "    # x: normalized float32 tensor [batch, 120, 251], on CPU.\n"
        "    step_logits, progress, mistake_logit, idle_logit = model(x)\n"
        "    return {\n"
        "        'step_logits': step_logits,        # [batch, 7], unrestricted real values\n"
        "        'step_scores': step_logits.sigmoid(),  # [batch, 7], independent [0, 1] scores\n"
        "        'idle_score': idle_logit.sigmoid(),\n"
        "        'mistake_score': mistake_logit.sigmoid(),\n"
        "        'progress': progress,\n"
        "    }\n"
        "```\n\n"
        "metrics.csv contains in-training metrics with dropout, not an independent test result or "
        "a final-checkpoint evaluation. No best checkpoint was selected on these metrics. "
        "Keep the original LOSO results for reporting generalization.\n", encoding="utf-8",
    )
    print(f"Deployment files saved: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--loso-run-dir", type=Path, default=DEFAULT_LOSO_RUN)
    parser.add_argument("--export-dir", type=Path, help="Default: DATA_ROOT/claude_l1_full")
    parser.add_argument("--output-dir", type=Path, help="Default: a new runs_3d_tune/L1_DEPLOY_* directory")
    parser.add_argument("--epochs", type=int, help="Override validation-epoch median (1..12)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "model_lstm/runs_3d_tune" / datetime.now().strftime("L1_DEPLOY_%Y-%m-%d_%H-%M-%S_%f")
    train_full_dataset(args.data_root, args.export_dir or args.data_root / "claude_l1_full",
                       output, args.loso_run_dir, args.epochs, args.seed, args.device)


if __name__ == "__main__":
    main()
