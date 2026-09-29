"""Claude L1 LSTM: prepare, train and evaluate the 15 participant-held-out folds.

Run from the repository root: python -B -m model_lstm.LSTM_loso_tune
Edit the paths in __main__; importing this module never starts training.
"""

from datetime import datetime
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data_proc_2d.app.build_norm_dataset_tune import prepare_l1_fold
from model_lstm.LSTM_databuilder_tune import AssistSequenceDataset, training_pos_weights
from model_lstm.LSTM_engine_tune import L1_SETTINGS, fit_model, run_epoch
from model_lstm.LSTM_model_train_tune import AssistLSTM, AssistLoss


def run_fold(data_root, export_root, run_root, test_uid, device, settings=None):
    config = {**L1_SETTINGS, **(settings or {})}
    fold_dir = prepare_l1_fold(data_root, export_root, test_uid)
    manifest = json.loads((fold_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    seed = manifest["fold_index"]
    cache_dir = config.get("dataset_cache_dir", Path(__file__).parent / ".dataset_cache" / "claude_l1" / f"fold_{test_uid:02d}")
    datasets, loaders = {}, {}
    for split, files in manifest["files"].items():
        datasets[split] = [AssistSequenceDataset(
            fold_dir / name, window_size=config["window_size"], stride=config["stride"],
            feature_keys=manifest["feature_keys"], cache_dir=cache_dir,
            expected_normalization_id=manifest["normalization_id"],
        ) for name in files]
        joined = ConcatDataset(datasets[split])
        if not len(joined):
            raise ValueError(f"No valid windows: fold {test_uid}, {split}")
        loaders[split] = DataLoader(
            joined, batch_size=config["batch_size"], shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(seed) if split == "train" else None,
            num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False,
        )
    if loaders["train"].dataset[0][0].shape[-1] != config["input_dim"]:
        raise ValueError("Input dimensions differ from the model configuration")
    pos_stats = training_pos_weights(datasets["train"], cap=config["pos_weight_cap"])
    torch.manual_seed(seed)
    model = AssistLSTM(**{key: config[key] for key in
                         ("input_dim", "hidden_dim", "num_steps", "dropout", "num_layers")}).to(device)
    criterion = AssistLoss(
        pos_stats["task_pos_weight"], pos_stats["mistake_pos_weight"],
        **{f"lambda_{key}": value for key, value in config["loss_weights"].items()},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    config.update(
        stage="claude_l1", test_uid=test_uid, fold_index=seed, seed=seed,
        feature_keys=manifest["feature_keys"], feature_profile=manifest["profile"],
        normalization_source=str(fold_dir / "norm_stats.npz"),
        normalization_id=manifest["normalization_id"],
        normalization_sha256=manifest["normalization_sha256"],
        pos_weight_stats=pos_stats, train_uids=manifest["subjects"]["train"],
        val_uids=manifest["subjects"]["val"], test_uids=manifest["subjects"]["test"],
        training_shuffle=True, shuffle_seed=seed, training_sampler="RandomSampler",
        sampler_replacement=False, sampler_num_samples=len(loaders["train"].dataset),
        task_target="task_id_vector[:7]", task_metric_filter="true_idle == 0",
        task_metric_truth="argmax(plateau[:7] + 0.001 * peak[:7])", macro_f1_labels=list(range(7)),
        idle_target="task_id_plateau_vector[7] >= 0.5",
        progress_mask="plateau[:7] >= 0.5", num_workers=0,
    )
    run_dir = Path(run_root) / f"fold_{test_uid:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fit_model(model, loaders["train"], loaders["val"], criterion, device, optimizer, run_dir, config)
    model.load_state_dict(torch.load(run_dir / "best_step_macro_f1.pth", map_location=device, weights_only=True))
    # Report the chosen checkpoint, with every training window evaluated once.
    train_eval = DataLoader(loaders["train"].dataset, batch_size=config["batch_size"], shuffle=False)
    for split, loader in (("train", train_eval), ("val", loaders["val"])):
        metrics, _ = run_epoch(model, loader, criterion, device)
        (run_dir / f"{split}_results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    results, predictions = run_epoch(model, loaders["test"], criterion, device, collect_predictions=True)
    predictions["file"] = np.concatenate([np.repeat(d.npz_path.name, len(d)) for d in datasets["test"]])
    predictions["target_frame"] = np.concatenate([d.target_slice for d in datasets["test"]])
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"test_uid": test_uid, "run_dir": str(run_dir), **results}


def run_loso(data_root, export_root, run_root, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    results = []
    for test_uid in range(1, 16):
        print(f"\nClaude L1 fold {test_uid}/15")
        results.append(run_fold(data_root, export_root, run_root, test_uid, device))
        (run_root / "fold_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {"protocol": "Claude L1", "folds": 15, "std_ddof": 0, "metrics": {}}
    for metric in ("step_macro_f1", "step_balanced_acc", "idle_f1", "mistake_f1",
                   "progress_mae_percent", "end_to_end_macro_f1", "end_to_end_acc"):
        values = [row[metric] for row in results if row[metric] is not None]
        summary["metrics"][metric] = dict(mean=float(np.mean(values)) if values else None,
                                          std=float(np.std(values)) if values else None,
                                          subjects=len(values))
    (run_root / "loso_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    data_root = Path(
        "G:/.shortcut-targets-by-id/1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC/"
        "ITECH_Thesis/Videos/dataset/skeleton_3d/ceiling_panel_installation_04"
    )
    run_root = Path(__file__).parent / "runs_3d_tune" / datetime.now().strftime("L1_LOSO_%Y-%m-%d_%H-%M-%S_%f")
    run_loso(data_root, data_root / "claude_l1", run_root)
