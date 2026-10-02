"""Shared train/eval loop; select checkpoints by complete eight-class macro-F1."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def class_scores(cm):
    cm = np.asarray(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(len(cm)), where=predicted != 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(len(cm)), where=support != 0)
    f1 = np.divide(2 * cm.diagonal(), support + predicted,
                   out=np.zeros(len(cm)), where=support + predicted != 0)
    return precision, recall, f1, support


def run_epoch(model, loader, device, optimizer=None, collect=False, gradient_clip=None):
    training = optimizer is not None
    model.train(training)
    names = ("step", "progress", "mistake", "background")
    loss_sums = torch.zeros(4, device=device, dtype=torch.float64)
    step_count = torch.zeros((), device=device, dtype=torch.long)
    progress_error = torch.zeros((), device=device, dtype=torch.float64)
    cm = torch.zeros(64, device=device, dtype=torch.long)
    task_cm = torch.zeros(49, device=device, dtype=torch.long)
    mistake_cm = torch.zeros(4, device=device, dtype=torch.long)
    windows, frames = 0, []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=device.type == "cuda")
            y = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in y.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss, parts, batch_steps = model.losses(out, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ablation loss")
            if training:
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            batch = len(x)
            windows += batch
            step_count += batch_steps
            loss_sums += torch.stack([parts[k].detach() * (batch_steps if k == "step" else batch)
                                      for k in names]).to(torch.float64)
            out = {k: v.detach() if v is not None else None for k, v in out.items()}
            pred, bg_score = model.decode(out)
            active = y["step"] != 7
            conditional = out["step"][:, :7].argmax(1)
            mistake = out["mistake"].argmax(1)
            cm += torch.bincount(8 * y["step"] + pred, minlength=64)
            task_cm += torch.bincount(7 * y["step"][active] + conditional[active], minlength=49)
            mistake_cm += torch.bincount(2 * y["mistake"] + mistake, minlength=4)
            progress_error += (out["progress"] - y["progress"]).abs().sum()
            if collect:
                frames.append(pd.DataFrame({
                    "true_step": y["step"].cpu().numpy(), "pred_step": pred.cpu().numpy(),
                    "pred_task_conditional": conditional.cpu().numpy(),
                    "true_background": (~active).long().cpu().numpy(),
                    "pred_background": (pred == 7).long().cpu().numpy(),
                    "background_score": bg_score.cpu().numpy(),
                    "true_progress": y["progress"].cpu().numpy(), "pred_progress": out["progress"].cpu().numpy(),
                    "true_mistake": y["mistake"].cpu().numpy(), "pred_mistake": mistake.cpu().numpy(),
                }))
    if not windows:
        raise ValueError("Empty data loader")
    cm, task_cm, mistake_cm = (v.cpu().numpy().reshape(size, size)
                              for v, size in ((cm, 8), (task_cm, 7), (mistake_cm, 2)))
    losses = loss_sums.cpu().numpy() / [max(int(step_count), 1), windows, windows, windows]
    scores = dict(zip(names, losses.tolist()))
    metrics = dict(
        loss=scores["step"] + scores["progress"] + scores["mistake"] + 0.2 * scores["background"],
        **{f"{k}_loss": v for k, v in scores.items()},
        macro_f1_8=float(class_scores(cm)[2].mean()), accuracy_8=float(cm.trace() / windows),
        background_f1=float(class_scores(cm)[2][7]),
        task_macro_f1_7_conditional=float(class_scores(task_cm)[2].mean()),
        task_accuracy_7_conditional=float(task_cm.trace() / max(task_cm.sum(), 1)),
        mistake_f1=float(class_scores(mistake_cm)[2][1]), mistake_accuracy=float(mistake_cm.trace() / windows),
        progress_mae_percent=100 * float(progress_error) / windows,
        windows=windows, non_background_windows=int(task_cm.sum()),
    )
    return metrics, cm, pd.concat(frames, ignore_index=True) if collect else None


def fit(model, train_loader, val_loader, device, run_dir, epochs, learning_rate=1e-4, weight_decay=0.01):
    run_dir = Path(run_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_f1, best_loss = -1.0, float("inf")
    for epoch in range(1, epochs + 1):
        train, _, _ = run_epoch(model, train_loader, device, optimizer)
        val, _, _ = run_epoch(model, val_loader, device)
        row = dict(epoch=epoch, **{f"train_{k}": v for k, v in train.items()},
                   **{f"val_{k}": v for k, v in val.items()})
        with (run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        if val["macro_f1_8"] > best_f1:
            best_f1 = val["macro_f1_8"]
            torch.save(model.state_dict(), run_dir / "best_macro_f1.pth")
            (run_dir / "best_macro_f1.json").write_text(json.dumps(dict(epoch=epoch, **val), indent=2))
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(model.state_dict(), run_dir / "best_val_loss.pth")
            (run_dir / "best_val_loss.json").write_text(json.dumps(dict(epoch=epoch, **val), indent=2))
        print(f"  Epoch {epoch:02d}/{epochs}: train loss {train['loss']:.4f}, "
              f"val macro-F1(8) {val['macro_f1_8']:.4f}, val loss {val['loss']:.4f}", flush=True)
    model.load_state_dict(torch.load(run_dir / "best_macro_f1.pth", map_location=device, weights_only=True))
