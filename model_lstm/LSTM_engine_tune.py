"""Shared L1 training/evaluation, used by the notebook and the LOSO runner."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


L1_SETTINGS = dict(
    model="LSTM", input_dim=251, hidden_dim=128, num_steps=7, num_layers=1,
    dropout=0.3, window_size=120, stride=10, stride_meaning="window_hop",
    sample_interval=1, predict_offset=0, mode="seq2one", batch_size=256, epochs=12,
    optimizer="AdamW", learning_rate=1e-3, weight_decay=1e-4,
    scheduler="CosineAnnealingLR", gradient_clip=1.0, pos_weight_cap=2.0,
    loss_weights=dict(step=1.0, progress=0.3, mistake=0.3, idle=0.2),
)


def run_epoch(model, data_loader, criterion, device, optimizer=None, collect_predictions=False, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals = dict(step=0.0, progress=0.0, mistake=0.0, idle=0.0)
    n_samples = n_active = n_progress = 0
    progress_abs_error = 0.0
    task_true, task_pred = [], []
    idle_true, idle_pred, mistake_true, mistake_pred = [], [], [], []
    full_true, full_pred = [], []
    frames = []
    with torch.set_grad_enabled(training):
        for x, y in data_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in y.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss, parts = criterion(outputs, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite multitask loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            step, progress, mistake, idle = (value.detach() for value in outputs)
            batch = len(x)
            active = y["plateau"] >= 0.5
            non_idle = y["idle"] < 0.5
            progress_mask = active & non_idle[:, None]
            active_count = int(active.sum())
            n_samples += batch
            n_active += active_count
            n_progress += int(progress_mask.sum())
            for name in ("step", "mistake", "idle"):
                totals[name] += float(parts[name].detach()) * batch
            totals["progress"] += float(parts["progress"].detach()) * active_count
            progress_abs_error += float((progress - y["progress"]).abs()[progress_mask].sum())

            truth = (y["plateau"] + 1e-3 * y["step"]).argmax(dim=1)
            prediction = step.argmax(dim=1)
            full_truth = torch.where(non_idle, truth, 7)
            full_prediction = torch.where(idle > 0, 7, prediction)
            full_true.extend(full_truth.cpu().tolist())
            full_pred.extend(full_prediction.cpu().tolist())
            task_true.extend(truth[non_idle].cpu().tolist())
            task_pred.extend(prediction[non_idle].cpu().tolist())
            idle_true.extend(y["idle"].cpu().tolist())
            idle_pred.extend((idle > 0).cpu().tolist())
            mistake_true.extend(y["mistake"].cpu().tolist())
            mistake_pred.extend((mistake > 0).cpu().tolist())
            if collect_predictions:
                data = {
                    "sample_index": np.arange(n_samples - batch, n_samples),
                    "true_step": torch.where(non_idle, truth, -1).cpu().numpy(),
                    "pred_step": prediction.cpu().numpy(),
                    "true_step_8": full_truth.cpu().numpy(),
                    "pred_step_8": full_prediction.cpu().numpy(),
                    "true_idle": y["idle"].cpu().numpy().astype(int),
                    "pred_idle": (idle > 0).cpu().numpy().astype(int),
                    "idle_prob": idle.sigmoid().cpu().numpy(),
                    "true_mistake": y["mistake"].cpu().numpy().astype(int),
                    "pred_mistake": (mistake > 0).cpu().numpy().astype(int),
                    "mistake_prob": mistake.sigmoid().cpu().numpy(),
                }
                for lane in range(7):
                    data[f"task_prob_{lane}"] = step[:, lane].sigmoid().cpu().numpy()
                    data[f"peak_{lane}"] = y["step"][:, lane].cpu().numpy()
                    data[f"plateau_{lane}"] = y["plateau"][:, lane].cpu().numpy()
                    data[f"true_progress_{lane}"] = y["progress"][:, lane].cpu().numpy()
                    data[f"pred_progress_{lane}"] = progress[:, lane].cpu().numpy()
                    data[f"progress_active_{lane}"] = progress_mask[:, lane].cpu().numpy()
                frames.append(pd.DataFrame(data))
    if n_samples == 0:
        raise ValueError("Empty loader")
    losses = {name: value / max(1, n_active if name == "progress" else n_samples)
              for name, value in totals.items()}
    metrics = {f"{name}_loss": value for name, value in losses.items()}
    metrics.update(
        loss=sum(criterion.weights[name] * value for name, value in losses.items()),
        step_acc=accuracy_score(task_true, task_pred) if task_true else None,
        step_macro_f1=f1_score(task_true, task_pred, labels=range(7), average="macro", zero_division=0) if task_true else None,
        step_balanced_acc=balanced_accuracy_score(task_true, task_pred) if task_true else None,
        idle_acc=accuracy_score(idle_true, idle_pred),
        idle_f1=f1_score(idle_true, idle_pred, zero_division=0),
        mistake_acc=accuracy_score(mistake_true, mistake_pred),
        mistake_f1=f1_score(mistake_true, mistake_pred, zero_division=0),
        progress_mae_percent=100 * progress_abs_error / n_progress if n_progress else None,
        end_to_end_macro_f1=f1_score(full_true, full_pred, labels=range(8), average="macro", zero_division=0),
        end_to_end_acc=accuracy_score(full_true, full_pred),
        windows=n_samples, task_windows=len(task_true),
        progress_loss_lanes=n_active, progress_metric_lanes=n_progress,
    )
    return metrics, pd.concat(frames, ignore_index=True) if frames else None

def fit_model(model, train_loader, val_loader, criterion, device, optimizer, run_dir, config):
    """Save both minima/maxima; validation task macro-F1 selects the test model."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.csv").exists():
        raise ValueError(f"Run already contains training results: {run_dir}")
    stats_source = Path(config["normalization_source"])
    (run_dir / "norm_stats.npz").write_bytes(stats_source.read_bytes())
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    best_loss, best_f1 = float("inf"), -1.0
    for epoch in range(1, config["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train, _ = run_epoch(model, train_loader, criterion, device, optimizer=optimizer,
                             gradient_clip=config["gradient_clip"])
        val, _ = run_epoch(model, val_loader, criterion, device)
        if val["step_macro_f1"] is None:
            raise ValueError("Validation needs non-idle windows for checkpoint selection")
        row = {"epoch": epoch, "learning_rate": learning_rate,
               **{f"train_{k}": v for k, v in train.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        with (run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {epoch}/{config['epochs']} | Train loss {train['loss']:.4f} "
              f"| Val loss {val['loss']:.4f} | Val task F1 {val['step_macro_f1']:.4f} "
              f"| Val idle F1 {val['idle_f1']:.4f}")
        summary = {"best_epoch": epoch, **val}
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if val["step_macro_f1"] > best_f1:
            best_f1 = val["step_macro_f1"]
            torch.save(model.state_dict(), run_dir / "best_step_macro_f1.pth")
            (run_dir / "summary_step_macro_f1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        scheduler.step()
    return run_dir
