# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import StepConditionedAssistTransformer


# ============================================================
# Dataset
# ============================================================
class AssistDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        self.X_cur = data["X_cur"].astype(np.float32)
        self.X_hist = data["X_hist"].astype(np.float32)  # can be empty M=0
        self.step_id = data["step_id"].astype(np.int64)

        self.element_feat = data["element_feat"].astype(np.float32)
        self.env_feat = data["env_feat"].astype(np.float32)
        self.pos_feat = data["pos_feat"].astype(np.float32)

        self.y_need = data["y_need"].astype(np.int64)
        self.y_type = data["y_type"].astype(np.int64)
        self.y_timing = data["y_timing"].astype(np.int64)

    def __len__(self):
        return len(self.X_cur)

    def __getitem__(self, idx):
        return {
            "X_cur": torch.tensor(self.X_cur[idx]),
            "X_hist": torch.tensor(self.X_hist[idx]),
            "step_id": torch.tensor(self.step_id[idx]),
            "element_feat": torch.tensor(self.element_feat[idx]),
            "env_feat": torch.tensor(self.env_feat[idx]),
            "pos_feat": torch.tensor(self.pos_feat[idx]),
            "y_need": torch.tensor(self.y_need[idx]),
            "y_type": torch.tensor(self.y_type[idx]),
            "y_timing": torch.tensor(self.y_timing[idx]),
        }


# ============================================================
# Train / Eval
# ============================================================
def train_one_epoch_loss(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total = 0

    for batch in loader:
        X_cur = batch["X_cur"].to(device)
        X_hist = batch["X_hist"].to(device)
        step_id = batch["step_id"].to(device)

        element_feat = batch["element_feat"].to(device)
        env_feat = batch["env_feat"].to(device)
        pos_feat = batch["pos_feat"].to(device)

        y_need = batch["y_need"].to(device)
        y_type = batch["y_type"].to(device)
        y_timing = batch["y_timing"].to(device)

        optimizer.zero_grad()

        logits_need, logits_type, logits_timing = model(
            X_cur=X_cur,
            X_hist=X_hist,
            step_id=step_id,
            element_feat=element_feat,
            env_feat=env_feat,
            pos_feat=pos_feat
        )

        loss_need = ce(logits_need, y_need)  # [B]
        loss_type = ce(logits_type, y_type)  # [B]
        loss_timing = ce(logits_timing, y_timing)  # [B]

        # only train type/timing when assist_needed=1
        mask = (y_need == 1).float()

        loss_type = (loss_type * mask).mean()
        loss_timing = (loss_timing * mask).mean()
        loss_need = loss_need.mean()

        loss = loss_need + 0.5 * loss_type + 0.5 * loss_timing

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_cur.size(0)
        total += X_cur.size(0)

    return total_loss / total


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()

    total = 0
    correct_need = 0

    for batch in loader:
        X_cur = batch["X_cur"].to(device)
        X_hist = batch["X_hist"].to(device)
        step_id = batch["step_id"].to(device)

        element_feat = batch["element_feat"].to(device)
        env_feat = batch["env_feat"].to(device)
        pos_feat = batch["pos_feat"].to(device)

        y_need = batch["y_need"].to(device)

        logits_need, _, _ = model(
            X_cur=X_cur,
            X_hist=X_hist,
            step_id=step_id,
            element_feat=element_feat,
            env_feat=env_feat,
            pos_feat=pos_feat
        )

        pred = logits_need.argmax(dim=-1)
        correct_need += (pred == y_need).sum().item()
        total += X_cur.size(0)

    return correct_need / total


# ============================================================
# Main
# ============================================================
def main():
    # ----------------------------
    # Config
    # ----------------------------
    npz_path = "dataset.npz"
    batch_size = 32
    lr = 1e-4
    epochs = 30

    # dims (must match dataset)
    pose_dim = 33 * 4      # example: mediapipe 33 joints * (x,y,z,vis)
    step_vocab = 20

    element_dim = 16
    env_dim = 8
    pos_dim = 8

    num_types = 5
    num_timing = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Data
    # ----------------------------
    dataset = AssistDataset(npz_path)

    # naive split
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = n - n_train

    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ----------------------------
    # Model
    # ----------------------------
    model = StepConditionedAssistTransformer(
        pose_dim=pose_dim,
        step_vocab=step_vocab,
        element_dim=element_dim,
        env_dim=env_dim,
        pos_dim=pos_dim,
        num_types=num_types,
        num_timing=num_timing,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=512
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ----------------------------
    # Train
    # ----------------------------
    best_val = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_loss(model, train_loader, optimizer, device)
        val_acc = eval_one_epoch(model, val_loader, device)

        print(f"[Epoch {epoch:03d}] loss={train_loss:.4f}  val_need_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print(f"  saved best model (val_need_acc={best_val:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
