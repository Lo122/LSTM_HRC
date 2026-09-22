"""Subject-disjoint data layer for the ceiling-panel recognition benchmark.

Everything the DATASET_CARD warns about is handled here, in one place, so every
model in the comparison sees exactly the same data and the same split:

  * split by uid (subject), never by file or segment -- all 3 camera views of a
    take are one performance and must stay on the same side
  * augmented copies join the TRAINING split only, and only for training uids
  * normalisation statistics computed on the TRAIN subjects only
  * azimuth encoded as sin/cos (it wraps at +/-180 and a network cannot know that)
  * ratios winsorised (they spike 8-10x when the lifter collapses shoulder width)
  * NaN frames masked out of windows rather than silently poisoning the LSTM

Label targets (see `labels.py`, which is the read-only source of truth):

  * `task_id` is an argmax with NO background guard -- on a frame where every lane
    is 0.0 (real idle time) it returns class 0, "Pull Cables". Measured corpus-wide,
    8.15% of frames are such background and 61.9% of all "Pull Cables" frames are
    actually idle. So we carry an explicit `bg` flag and never train the task head
    on a background frame's phantom label.
  * `task_id_plateau_vector` is flat 1.0 across each annotated span -- the correct
    multi-label BCE target. `task_id_vector` (asymmetric_peak) peaks only at the
    span midpoint and is an argmax tie-breaker, not a span-presence signal.
"""
from __future__ import annotations
import glob, os, re, json
import numpy as np
import torch

PANEL_ORDER = [
    "joint_speed", "joint_acceleration",
    "joint_velocity_x", "joint_velocity_y", "joint_velocity_z",
    "joint_acceleration_x", "joint_acceleration_y", "joint_acceleration_z",
    "position_x_relative_to_pelvis", "position_y_relative_to_pelvis",
    "position_z_relative_to_pelvis",
    "polar_azimuth", "polar_elevation", "joint_angles", "ratios",
    "distance_from_center",
]
ALL_TASK_NAMES = ["Pull Cables", "Lift", "Align", "Screw", "Connect", "Clamp", "Place"]
LIFT_LANE = 1

# Lift is an umbrella span covering the whole carry-and-hang sequence, with Place
# (94% nested) and Align (86%) living inside it. Two measured consequences:
#   * it is the lane the model hallucinates -- predicted >=0.5 on 73% of frames
#     where it is genuinely absent, which is why gating the trigger on it fails
#   * dropping it costs almost nothing: only 3.2% of frames have Lift as their
#     ONLY active class, and those correctly become background
# Kept as a switch so the 7-class arms stay reproducible for the thesis table.
DROP_LIFT = True


def task_names(drop_lift=None):
    d = DROP_LIFT if drop_lift is None else drop_lift
    return ([n for i, n in enumerate(ALL_TASK_NAMES) if i != LIFT_LANE]
            if d else list(ALL_TASK_NAMES))


def n_tasks(drop_lift=None):
    return len(task_names(drop_lift))


def _keep_lanes(drop_lift):
    return ([i for i in range(len(ALL_TASK_NAMES)) if i != LIFT_LANE]
            if drop_lift else list(range(len(ALL_TASK_NAMES))))


# Module-level defaults, kept for callers that do not thread the flag through.
TASK_NAMES = task_names()
N_TASKS = n_tasks()


def take_key(path: str) -> tuple[int, str]:
    """(uid, take) -- the grouping key. Camera is deliberately NOT part of it."""
    b = os.path.basename(path)
    uid = int(re.search(r"uid-(\d+)", b).group(1))
    take = re.search(r"take-([\d-]+?)(?:_aug|_seg|\.pt)", b).group(1)
    return uid, take


def cam_take_key(path: str) -> tuple[str, int, str]:
    """(cam, uid, take) -- identifies one physical recording, for dedup."""
    b = os.path.basename(path)
    cam = re.search(r"cam-(\d+)", b).group(1)
    uid, take = take_key(path)
    return cam, uid, take


def dedup_augmented(paths):
    """The mirror copies _aug-01/02/03 of a take are byte-identical: mirroring is
    deterministic and the varying seeds only drive `rotation`/`noise`, both None
    in this corpus. Keeping all three would weight mirrored samples 3x against
    originals, so keep exactly one per physical recording."""
    seen, out = set(), []
    for p in sorted(paths):
        k = cam_take_key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def load_take(path: str, drop_lift=None):
    """-> dict of arrays, all aligned on the frame axis.

    X          (T, 251) float32   251 = 235 raw cols, azimuth's 16 deg -> 32 sin/cos
    task       (T,)     int64     argmax label (background-contaminated, see module doc)
    plateau    (T, C)   float32   flat-top per-class scores -- the BCE target
    prog_vec   (T, C)   float32   per-lane progress, 0-1
    mistake    (T,)     int64
    prog       (T,)     float32   scalar progress of the argmax lane, 0-1
    bg         (T,)     bool      True where NO task is annotated
    valid      (T,)     bool      finite features
    """
    drop_lift = DROP_LIFT if drop_lift is None else drop_lift
    keep = _keep_lanes(drop_lift)
    d = torch.load(path, map_location="cpu", weights_only=False)
    f, L = d["features"], d["labels"]
    cols = []
    for k in PANEL_ORDER:
        a = f[k].numpy().astype(np.float32)
        if k == "polar_azimuth":                      # wraps at +/-180 -> sin/cos
            r = np.deg2rad(a)
            cols += [np.sin(r), np.cos(r)]
        elif k == "ratios":                           # shared-denominator spikes
            cols.append(np.clip(a, 0.0, 4.0))
        else:
            cols.append(a)
    X = np.concatenate(cols, axis=1)

    plateau = L["task_id_plateau_vector"].numpy().astype(np.float32)[:, keep]
    # Exact background detector: verified to agree with (task_id_prob == 0) on
    # 100.00% of frames. The int task_id cannot express this -- it says class 0.
    # Computed AFTER dropping lanes, so the 3.2% of frames whose only active
    # class was Lift correctly become background instead of silently vanishing.
    bg = plateau.max(axis=1) <= 0.0

    return {
        "X": X,
        "task": L["task_id"].numpy().astype(np.int64),
        "plateau": plateau,
        # Eval-only tie-breaker. The plateau curve saturates at exactly 1.0, so
        # 18% of windows have two or more lanes tied at the max and a plain
        # argmax resolves every one of them toward the lowest class index --
        # which erases Place (index 6) entirely (9.6% of windows -> 0.5%).
        # The peak curve exists precisely to break these ties (labels.py:288-291).
        "peak": L["task_id_vector"].numpy().astype(np.float32)[:, keep],
        "prog_vec": L["task_progress_vector"].numpy().astype(np.float32)[:, keep] / 100.0,
        "mistake": L["mistake"].numpy().astype(np.int64),
        "prog": L["task_progress"].numpy().astype(np.float32) / 100.0,
        "bg": bg,
        "valid": np.isfinite(X).all(axis=1),
    }


def subject_split(paths, test_uids, val_uids, aug_paths=None):
    """Split by subject. Augmented files, if given, join TRAIN only and only for
    uids that are already in train -- a mirrored copy of a held-out subject would
    be a direct leak."""
    tr = [p for p in paths if take_key(p)[0] not in test_uids | val_uids]
    va = [p for p in paths if take_key(p)[0] in val_uids]
    te = [p for p in paths if take_key(p)[0] in test_uids]
    if aug_paths:
        held = test_uids | val_uids
        tr = tr + [p for p in dedup_augmented(aug_paths)
                   if take_key(p)[0] not in held]
    return tr, va, te


class WindowSet:
    """Causal sliding windows. Every label is read at the window's LAST frame, so
    each window is a legitimate real-time prediction point."""

    def __init__(self, paths, win=120, hop=10, stats=None, drop_lift=None):
        self.win, self.hop = win, hop
        self.drop_lift = DROP_LIFT if drop_lift is None else drop_lift
        self.task_names = task_names(self.drop_lift)
        self.Xs, self.idx = [], []
        self.task, self.plateau, self.prog_vec = [], [], []
        self.mistake, self.prog, self.bg, self.peak = [], [], [], []
        for p in paths:
            d = load_take(p, self.drop_lift)
            ti = len(self.Xs)
            self.Xs.append(d["X"])
            self.task.append(d["task"]); self.plateau.append(d["plateau"])
            self.prog_vec.append(d["prog_vec"]); self.mistake.append(d["mistake"])
            self.prog.append(d["prog"]); self.bg.append(d["bg"])
            self.peak.append(d["peak"])
            # a window is usable only if every one of its frames has features
            c = np.concatenate([[0], np.cumsum(d["valid"])])
            for e in range(win, len(d["X"]) + 1, hop):
                if c[e] - c[e - win] == win:
                    self.idx.append((ti, e))
        self.idx = np.array(self.idx, dtype=np.int64)

        if stats is None:                              # TRAIN ONLY
            sample = np.concatenate([X[::7] for X in self.Xs])
            sample = sample[np.isfinite(sample).all(1)]
            mu, sd = sample.mean(0), sample.std(0)
            self.stats = (mu.astype(np.float32), (sd + 1e-6).astype(np.float32))
        else:
            self.stats = stats

    def __len__(self):
        return len(self.idx)

    def tensors(self):
        """Materialise the whole set. Returns (X, y) where y is a dict of targets."""
        mu, sd = self.stats
        N, D = len(self.idx), self.Xs[0].shape[1]
        out = np.empty((N, self.win, D), dtype=np.float32)
        for i, (ti, e) in enumerate(self.idx):
            out[i] = (self.Xs[ti][e - self.win:e] - mu) / sd
        ti, e = self.idx[:, 0], self.idx[:, 1] - 1     # label at last frame
        gather = lambda src: np.array([src[a][b] for a, b in zip(ti, e)])
        y = {
            "task": torch.from_numpy(gather(self.task)),
            "plateau": torch.from_numpy(gather(self.plateau)),
            "prog_vec": torch.from_numpy(gather(self.prog_vec)),
            "mistake": torch.from_numpy(gather(self.mistake)),
            "prog": torch.from_numpy(gather(self.prog)),
            "bg": torch.from_numpy(gather(self.bg).astype(np.float32)),
            "peak": torch.from_numpy(gather(self.peak)),
        }
        return torch.from_numpy(out), y

    def take_index(self):
        """Which take each window came from -- so smoothing never crosses takes."""
        return self.idx[:, 0]
