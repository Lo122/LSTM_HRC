import numpy as np
import torch
from torch.utils.data import Dataset

# load data from frame-wise to sequence-wise for LSTM training
# choose seq2one first > only needs to predict the last frame's label, which is more stable and easier to train

class AssistSequenceDataset(Dataset):
    def __init__(
        self,
        npz_path,
        window_size=120,       # how many frames as input (e.g., 4 seconds at 30fps)
        stride=30,             # stride to slide the window (e.g., 1 second at 30fps)
        predict_offset=0,      # predict how many frames ahead (e.g., 0 for current frame, 30 for 1 second later)
        mode="seq2one",        # "seq2one" or "seq2seq"
        use_type=False         # predict assist type or not
    ):
        super().__init__()

        data = np.load(npz_path)

        self.deg = data["X_degree"]      # [T, pose_dim]
        self.speed = data["X_speed"]      # [T, step_dim]

        self.step = data["X_step"]      # [T, step_dim]
        
        self.window_size = window_size
        self.stride = stride
        self.predict_offset = predict_offset
        self.mode = mode
        self.use_type = use_type

        self.T = len(self.deg)  # total number of frames

        # ===== construct sample index  =====
        self.indices = []
        for start in range(0, self.T - window_size - predict_offset + 1, stride):
            self.indices.append(start)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        start = self.indices[idx]
        end = start + self.window_size
        target_index = end - 1 + self.predict_offset

        # ===== concatenate features =====
        x_deg = self.deg[start:end]
        x_speed = self.speed[start:end]
        # x_elem = self.elem[start:end]
        # x_env  = self.env[start:end]

        # x = np.concatenate([x_pose, x_elem, x_env], axis=1)
        x = np.hstack([x_deg, x_speed])  # concatenate degree and speed features
        x = x.astype(np.float32)  # ensure it's float32 for PyTorch
        x = torch.tensor(x, dtype=torch.float32)

        # ===== construct label =====
        if self.mode == "seq2one":

            y_step = self.step[target_index]
            y = torch.tensor(y_step, dtype=torch.float32)

        elif self.mode == "seq2seq":


            y_step = self.step[start:end]
            y = torch.tensor(y_step, dtype=torch.float32)

        else:
            raise ValueError("mode must be seq2one or seq2seq")

        return x, y