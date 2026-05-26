import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_KEYS = [
    "velocity_xy",
    "acceleration_xy",
    "pol_vectors",
    "pol_angles",
    "joint_angles",
    "ratios",
    "dist_ratios",
    "angles_combined",
]


class AssistSequenceDataset(Dataset):
    def __init__(
        self,
        npz_path,
        window_size=120,
        stride=30,
        predict_offset=0,
        mode="seq2one",   # "seq2one" or "seq2seq"
        feature_keys=None # choose features here
    ):
        super().__init__()

        data = np.load(npz_path)

        # ===== feature selection =====
        if feature_keys is None:
            feature_keys = ["angles_combined"] # the combo feature

        self.feature_keys = feature_keys

        # load selected features
        self.features = [data[k] for k in feature_keys]

        # ===== labels =====
        self.step = data["step_id"]            # (T,)
        self.step_vec = data["step_id_vector"] # (T,7)
        self.status = data["status_id"]
        self.progress = data["task_progress"]

        # ===== config =====
        self.window_size = window_size
        self.stride = stride
        self.predict_offset = predict_offset
        self.mode = mode

        self.T = self.features[0].shape[0]

        # ===== build indices =====
        self.indices = []
        for start in range(0, self.T - window_size - predict_offset + 1, stride):
            self.indices.append(start)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        start = self.indices[idx]
        end = start + self.window_size
        target_index = end - 1 + self.predict_offset

        # =====================================================
        # build X (dynamic feature concat)
        # =====================================================
        feat_list = []

        for f in self.features:
            feat_list.append(f[start:end])  # (window, dim)

        x = np.concatenate(feat_list, axis=1)  # (window, total_dim)
        x = torch.tensor(x, dtype=torch.float32)

        # =====================================================
        # build Y
        # =====================================================
        if self.mode == "seq2one":
            y = torch.tensor(self.step[target_index], dtype=torch.long)

        elif self.mode == "seq2seq":
            y = torch.tensor(self.step[start:end], dtype=torch.long)

        else:
            raise ValueError("mode must be seq2one or seq2seq")

        return x, y