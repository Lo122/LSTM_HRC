# train_rnn.py
# Train GRU/LSTM baselines on dataset.npz

import os
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from model_rnn import StepConditionedAssistRNN


# ============================================================
# Dataset
# ============================================================
class AssistDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.X_main = data["X_main"].astype(np.float32)
        self.X_hist = data["X_hist"].astype(np.float32)
        self.X_step = data["X_step"].astype(np.int64)
        self.X_elem = data["X_elem"].astype(np.float32)
        self.X_env = data["X_env"].astype(np.float32)
        self.X_pos = data["X_pos"].astype(np.float32)

        self.Y_need = data["Y_need"].astype(np.int64)
        self.Y_type = data["Y_type"].astype(np.int64)
        self.Y_timing = data["Y_timing"].astype(np.int64)

        # dims
        self.pose_dim = int(data["pose_dim"])
        self.elem_dim = int(data["elem_dim"])
        self.env_dim = int(data["env_dim"])
        self.pos_dim = int(data["pos_dim"])

    def __len__(self):
        return len(self.X_main)

    def __getitem__(self, idx):
        return {
            "X_main": torch.from_numpy(self.X_main[idx]),
            "X_hist": torch.from_numpy(self.X_hist[idx]),
            "X_step": torch.tensor(self.X_step[idx], dtype=torch.long),
            "X_elem": torch.from_numpy(self.X_elem[idx]),
            "X_env": torch.from_numpy(self.X_env[idx]),
            "X_pos": torch.from_numpy(self.X_pos[idx]),
            "Y_need": torch.tensor(self.Y_need[idx], dtype=torch.long),
            "Y_type": torch.tensor(self.Y_type[idx], dtype=torch.long),
            "Y_timing": torch.tensor(self.Y_timing[idx], dtype=torch.long),
        }


# ============================================================
# Train / Eval
# ============================================================
def accuracy(logits, y):
    pred = torch.argmax(logits, dim=-1)
    return (pred == y).float().mean().item()


def train_one_epoch(model, loader, optim, device, loss_w=(1.0, 1.0, 1.0)):
    model.train()

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_need = 0.0
    total_type = 0.0
    total_timing = 0.0

    for batch in loader:
        X_main = batch["X_main"].to(device)
        X_hist = batch["X_hist"].to(device)
        X_step = batch["X_step"].to(device)
        X_elem = batch["X_elem"].to(device)
        X_env = batch["X_env"].to(device)
        X_pos = batch["X_pos"].to(device)

        Y_need = batch["Y_need"].to(device)
        Y_type = batch["Y_type"].to(device)
        Y_timing = batch["Y_timing"].to(device)

        need_logits, type_logits, timing_logits = model(
            X_main=X_main,
            X_hist=X_hist,
            X_step=X_step,
            X_elem=X_elem,
            X_env=X_env,
            X_pos=X_pos,
        )

        loss_need = ce(need_logits, Y_need)
        loss_type = ce(type_logits, Y_type)
        loss_timing = ce(timing_logits, Y_timing)

        loss = loss_w[0] * loss_need + loss_w[1] * loss_type + loss_w[2] * loss_timing

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total_loss += loss.item()
        total_need += accuracy(need_logits, Y_need)
        total_type += accuracy(type_logits, Y_type)
        total_timing += accuracy(timing_logits, Y_timing)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "acc_need": total_need / n,
        "acc_type": total_type / n,
        "acc_timing": total_timing / n,
    }


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_need = 0.0
    total_type = 0.0
    total_timing = 0.0

    for batch in loader:
        X_main = batch["X_main"].to(device)
        X_hist = batch["X_hist"].to(device)
        X_step = batch["X_step"].to(device)
        X_elem = batch["X_elem"].to(device)
        X_env = batch["X_env"].to(device)
        X_pos = batch["X_pos"].to(device)

        Y_need = batch["Y_need"].to(device)
        Y_type = batch["Y_type"].to(device)
        Y_timing = batch["Y_timing"].to(device)

        need_logits, type_logits, timing_logits = model(
            X_main=X_main,
            X_hist=X_hist,
            X_step=X_step,
            X_elem=X_elem,
            X_env=X_env,
            X_pos=X_pos,
        )

        loss_need = ce(need_logits, Y_need)
        loss_type = ce(type_logits, Y_type)
        loss_timing = ce(timing_logits, Y_timing)
        loss = loss_need + loss_type + loss_timing

        total_loss += loss.item()
        total_need += accuracy(need_logits, Y_need)
        total_type += accuracy(type_logits, Y_type)
        total_timing += accuracy(timing_logits, Y_timing)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "acc_need": total_need / n,
        "acc_type": total_type / n,
        "acc_timing": total_timing / n,
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="dataset.npz")
    ap.add_argument("--outdir", type=str, default="checkpoints_rnn")
    ap.add_argument("--rnn_type", type=str, default="gru", choices=["gru", "lstm"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--bidir", action="store_true")
    ap.add_argument("--use_history", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds = AssistDataset(args.npz)
    n = len(ds)
    n_train = int(n * 0.8)
    n_val = n - n_train

    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # IMPORTANT: num_steps can be larger than your real step_id range
    model = StepConditionedAssistRNN(
        pose_dim=ds.pose_dim,
        elem_dim=ds.elem_dim,
        env_dim=ds.env_dim,
        pos_dim=ds.pos_dim,
        num_steps=64,
        num_types=5,
        num_timings=3,
        rnn_type=args.rnn_type,
        rnn_hidden=args.hidden,
        rnn_layers=args.layers,
        bidirectional=args.bidir,
        dropout=0.25,
        use_history=args.use_history,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, device)
        va = eval_one_epoch(model, val_loader, device)
        dt = time.time() - t0

        msg = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "sec": dt,
        }
        history.append(msg)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {tr['loss']:.3f} need {tr['acc_need']:.3f} type {tr['acc_type']:.3f} timing {tr['acc_timing']:.3f} | "
            f"val loss {va['loss']:.3f} need {va['acc_need']:.3f} type {va['acc_type']:.3f} timing {va['acc_timing']:.3f} | "
            f"{dt:.1f}s"
        )

        # choose a main metric (need accuracy)
        score = va["acc_need"]

        if score > best_val:
            best_val = score
            ckpt = {
                "model_state": model.state_dict(),
                "config": vars(args),
                "pose_dim": ds.pose_dim,
                "elem_dim": ds.elem_dim,
                "env_dim": ds.env_dim,
                "pos_dim": ds.pos_dim,
            }
            torch.save(ckpt, os.path.join(args.outdir, f"best_{args.rnn_type}.pt"))
            print(f"[SAVE] best checkpoint updated: {best_val:.4f}")

    # save training log
    with open(os.path.join(args.outdir, f"trainlog_{args.rnn_type}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nDone.")
    print("Best val acc_need:", best_val)


if __name__ == "__main__":
    main()