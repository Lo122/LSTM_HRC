# LSTM training-data pipeline (3D)

Turns raw task videos into the segmented `{features, labels}` tensors the
LSTM trains on. Four scripts, run in order:

```
raw video (cam-04/*.mp4)
      │
      ▼
generate_lstm_training_data.py   YOLO 2D pose → MotionBERT 2D→3D lift → bone-length stabilize
      │
      ▼  video__cam-04_uid-XX_take-XX.npz   (keypoints_3d: (T,17,3) root-relative H36M skeleton, meters)
      │
      ├─────────────────────────────┐
      ▼                              ▼
build_training_pairs.py       augment_dataset.py   (rotation + joint-noise copies)
      │                              │
      ▼  features__..._take-XX.pt    ▼  features__..._take-XX_aug-NN.pt
      │        (dataset/original/)        (dataset/augmented/)
      └──────────────┬───────────────┘
                      ▼
              segment_data.py   fixed-length (SEGMENT_SIZE-frame) slices
                      │
                      ▼
              dataset/segment/, dataset/augmented_segment/   ← LSTM training input
```

Each `.pt` file is a `torch.save`d dict: `{"metadata": ..., "features": {panel_name: (T, n_cols) tensor, ...}, "labels": {...}}`.

This is the 3D counterpart of `data_proc_2d/app/`, following the same
`app/` (thin CLI orchestrator) vs `src/` (logic, in
`data_proc_3d/src/skeleton_pipeline/`) split. The 2D pipeline reads
MediaPipe/YOLO 2D landmarks directly; this one lifts them to 3D with
MotionBERT first. Everything downstream is otherwise analogous —
`data_proc_2d/app/build_training_pairs.py` and
`data_proc_2d/app/segment_data.py` are the 2D siblings of the last two
steps below.

## 1. `generate_lstm_training_data.py` — video → 3D skeleton `.npz`

For every video in `--video-dir`:

1. **YOLO** 2D pose detection per frame.
2. Optional `KeypointOutlierHoldFilter` — holds the last good keypoint
   through single-frame detector glitches.
3. **MotionBERT** (`DSTformer`) lifts the rolling window of 2D keypoints to
   a root-relative 3D H36M skeleton. `--clip-len` defaults to **81**
   (matching the live/deployment window), not the higher-quality 243 —
   training on 243 would give the LSTM a systematically
   smoother/less-jittery input distribution than it sees live.
4. `BoneLengthConstraintFilter`, reset per video, stabilizes bone lengths
   over that one clip — removes frame-to-frame shrink/stretch noise from
   monocular depth ambiguity *within* a subject, without rescaling
   *between* subjects (each video keeps its own converged bone lengths;
   see the script's docstring for why no cross-video "unit skeleton"
   normalization is applied — it would erase a real between-subject height
   signal the LSTM may want to use).
5. Optional `--gravity-align` — rotates the skeleton by the camera's
   calibrated extrinsics so "up" is true gravity-up, not the camera's own
   (possibly tilted) axis. Requires `--calib-dir` extrinsics; skipped with
   a warning otherwise.

**Output** (`<output-dir>/video__<take-id>.npz`): `keypoints_2d`,
`keypoints_3d`, `timestamps`, `fps`, `bone_length_edges` /
`bone_length_targets` (the stabilizer's converged per-bone lengths),
`body_scale_m` (a convenience leg+spine height proxy derived from those
lengths), `gravity_aligned`. Also writes, per video, a `_render.mp4`
(2D overlay | 3D 4-view skeleton) and per-feature-group plots, unless
`--no-render-video` / `--no-plots`.

```powershell
uv run python generate_lstm_training_data.py `
    --video-dir "G:\...\Videos\raw\cam-04" `
    --output-dir "...\data_proc_3d\results" `
    --device cuda:0 `
    --motionbert-config "...\MotionBERT\configs\pose3d\MB_ft_h36m.yaml" `
    --motionbert-checkpoint "...\MotionBERT\checkpoint\pose3d\FT_MB_release_MB_ft_h36m\best_epoch.bin"
```

## 2. `build_training_pairs.py` — `.npz` + annotation `.json` → `.pt`

Per take, does two things and saves them together:

- **Features**: `skeleton_pipeline.dataset.feature_io.extract_features`
  computes kinematic features (joint speeds/angles/distances, etc. — see
  `skeleton_pipeline/features/h36m_features.py`) from the raw `(T,17,3)`
  positions.
- **Labels**: `skeleton_pipeline.dataset.labels.extract_labels` reads the
  matching `label__<take-id>.json` / `label_detail__<take-id>.json` ELAN
  annotation pair (same files `data_proc_2d/app/proc_annotations.py`
  produces) and turns them into smooth per-frame label tensors
  (`step_id`, `status_id`, `task_progress`, plateau variants, etc.).

Both are written together as one `{"metadata", "features", "labels"}` dict
to `dataset/original/features__<take-id>.pt`.

> This file used to be named `extract_features.py` — renamed because that
> name only described the feature-computation half of what it does; see
> [Renamed files](#renamed-files) below.

```powershell
uv run python build_training_pairs.py
```

## 3. `augment_dataset.py` — rotation + joint-noise copies

For every `video__<take-id>.npz` that `build_training_pairs.py` also
reads, writes `--num-augmentations` (default 3) augmented copies —
`features__<take-id>_aug-01.pt`, `_aug-02.pt`, ... — to
`dataset/augmented/`, each with an independent random draw of:

- **rotation** — about `--rotation-axis` (default `z`, yaw-only: rotating
  a standing human about anything but the vertical axis produces
  physically implausible poses), and
- **noise** — per-joint, per-frame Gaussian jitter (`--noise-sigma-m`,
  default 1 cm).

Both transforms run on the *raw* `(T,17,3)` positions, then features are
recomputed from the augmented positions. Labels are **not** recomputed —
a spatial transform of the skeleton doesn't change step/status/progress,
so the same label tensors `build_training_pairs.py`'s
`labels.extract_labels` produces are reused as-is for every augmented copy
of a take.

Run this alongside (or after) `build_training_pairs.py` — they're
independent, and a typical training set uses both the un-augmented
"original" takes and these augmented ones.

```powershell
uv run python augment_dataset.py
uv run python augment_dataset.py --num-augmentations 5 --noise-sigma-m 0.02 --rotation-axis z
uv run python augment_dataset.py --num-augmentations 5 --no-noise --rotation-axis z
```

## 4. `segment_data.py` — fixed-length segments

Splits every `{"metadata", "features", "labels"}` .pt take under
`--in-dir` into fixed-length, `--segment-size`-frame (default 1000)
segments, via `skeleton_pipeline.dataset.segment` (dtype-agnostic slicing
— works unmodified on both `build_training_pairs.py`'s "original" takes
and `augment_dataset.py`'s "augmented" ones; point `--in-dir` at either).

```powershell
uv run python segment_data.py
uv run python segment_data.py --in-dir ../results/dataset/augmented --out-dir ../results/dataset/augmented_segment
```

Run it twice — once per source directory — to get both
`dataset/segment/` (from `original/`) and `dataset/augmented_segment/`
(from `augmented/`) as final LSTM training input.

## Data reference

### `.npz` (per-video, `generate_lstm_training_data.py` output)

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `keypoints_2d` | `(T, 17, 2)` | Raw YOLO 2D detections (pixels). |
| `keypoints_3d` | `(T, 17, 3)` | Root-relative H36M skeleton, meters, bone-length-stabilized. `NaN` where no person was detected. |
| `timestamps` | `(T,)` | Seconds from video start (`frame_idx / fps`). |
| `fps` | scalar | Source video frame rate. |
| `bone_length_edges` / `bone_length_targets` | `(E, 2)` / `(E,)` | Converged per-bone lengths from `BoneLengthConstraintFilter`, this video only. |
| `body_scale_m` | scalar | Leg+spine chain sum from the above — a convenience per-subject scale proxy, not a calibrated real height. |
| `gravity_aligned` | bool | Whether `--gravity-align` was applied. |

### `.pt` (per take, `build_training_pairs.py` / `augment_dataset.py` output)

`{"metadata": {...}, "features": {...}, "labels": {...}}`, `torch.save`d.
`features` and `labels` are flat `{column_name: (T,) or (T, n) tensor}`
dicts, not nested by panel — `metadata["panel_columns"]` records which
feature columns belong to which panel (see table below), for plotting/
grouping.

- **`metadata`** — `total_frames`, `fps`, the source `.npz`'s
  `bone_length_edges`/`bone_length_targets`/`body_scale_m`/
  `gravity_aligned`, `panel_columns` (feature column names per panel),
  `label_config`, and — for augmented copies only — the `augmentation`
  params actually drawn (rotation angle/axis, noise sigma).

#### `features` — `skeleton_pipeline/features/h36m_features.py`

All 16 non-pelvis H36M joints (`r_hip, r_knee, r_ankle, l_hip, l_knee,
l_ankle, spine, thorax, neck, head, l_shoulder, l_elbow, l_wrist,
r_shoulder, r_elbow, r_wrist`) — pelvis itself is skipped everywhere since
skeletons are root-relative (pelvis pinned at the origin, so its own
features are trivially zero). Everything computed directly from position
(angles/polar/ratios/distance-from-center) uses a Savitzky-Golay-smoothed
position, not the raw noisy one; velocity/acceleration are the 1st/2nd
analytic derivatives of that same filter — see the module docstring for
why (a plain frame-difference would turn single bad-lifter frames into
huge spikes).

| Panel | Columns (`{joint}_...`) | Count | Units | What it is |
| --- | --- | --- | --- | --- |
| Joint Speed | `{joint}_speed` | 16 | m/s | \|velocity\| magnitude, per joint |
| Joint Acceleration | `{joint}_acceleration` | 16 | m/s² | \|acceleration\| magnitude, per joint |
| Joint Velocity X/Y/Z | `{joint}_velocity_{x,y,z}` | 3×16 | m/s | Per-axis velocity components |
| Joint Acceleration X/Y/Z | `{joint}_acceleration_{x,y,z}` | 3×16 | m/s² | Per-axis acceleration components |
| Position X/Y/Z (relative to pelvis) | `{joint}_position_{x,y,z}` | 3×16 | m | Smoothed root-relative position |
| Polar Azimuth | `{joint}_azimuth_deg` | 16 | deg | Direction of the joint in the horizontal (XY) plane, `atan2(y, x)` |
| Polar Elevation | `{joint}_elevation_deg` | 16 | deg | Angle above/below horizontal, `atan2(z, sqrt(x²+y²))` |
| Joint Angles | `left_elbow_angle_deg`, `right_elbow_angle_deg`, `left_shoulder_angle_deg`, `right_shoulder_angle_deg`, `left_hip_angle_deg`, `right_hip_angle_deg`, `left_knee_angle_deg`, `right_knee_angle_deg`, `neck_angle_deg` | 9 | deg | 3-point angle at each named joint between its two adjacent bone vectors |
| Ratios | `elbow_over_shoulder_ratio`, `wrist_over_shoulder_ratio` | 2 | unitless | Mean(left, right) distance of that joint from the shoulder midpoint, ÷ shoulder width — a scale-invariant "how far the arm reaches" measure |
| Distance from Center | `{joint}_distance_from_center_ratio` | 16 | unitless | Joint's distance from the pelvis, ÷ torso height (pelvis→thorax) — scale-invariant across subjects |

> `data_proc_2d/results/.../feature_results.json` documents the original
> 2D panel/feature-name schema this mirrors; the 2D source that produced
> it (`feature_extraction.py`/`feature_analysis.py`) no longer exists in
> this repo (only unrecoverable `.pyc` remains), so this 3D module is a
> fresh, documented reimplementation matching those names/groupings, not
> a guaranteed-numerically-identical port — see the module docstring.

#### `labels` — `skeleton_pipeline/dataset/labels.py`

Built from two ELAN-derived annotation files per take,
`label__<take-id>.json` (coarse **step**) and
`label_detail__<take-id>.json` (fine-grained **status**, with
per-piece progress checkpoints), each a list of
`{step_id, piece_id, start_frame, end_frame, [start_progress,
end_progress]}` entries. `step_id` here is one of:

| id | step | id | step |
| --- | --- | --- | --- |
| 0 | Place Spacer | 4 | Place the Piece |
| 1 | Move Spacer | 5 | Screw |
| 2 | Remove spacer | 6 | Mistake |
| 3 | Align the Piece | 7–11 | `... - BL` baseline variants — **excluded** from training |

Each entry becomes a smooth [0, 1] per-frame score curve, ramped in/out
over a `buffer` of 100 frames around `[start_frame, end_frame]` with a
cubic smoothstep ("bezier") ease — see `LabelConfiguration` /
`_define_step_label_entry`. Two ramp shapes are built for both the step
and status entries:

- **asymmetric_peak** — ramps meet at a single apex (the span's temporal
  midpoint) instead of a flat top, so `idxmax` never ties across a whole
  plateau when picking one winning label per frame.
- **plateau** — flat 1.0 for the whole `[start_frame, end_frame]` span,
  smoothstep ramps only at the edges.

| Column | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `step_id` | `(T,)` | int64 | Coarse step with the highest asymmetric-peak score this frame (`-1` if no annotation covers it) |
| `step_id_prob` | `(T,)` | float32 | That winning step's score, `[0, 1]` |
| `step_id_plateau` / `step_id_plateau_prob` | `(T,)` | int64 / float32 | Same, using the flat-top plateau curve instead |
| `status_id` / `status_id_prob` | `(T,)` | int64 / float32 | Same as `step_id`/`_prob` but built from `label_detail`'s (fine-grained status) entries |
| `status_id_plateau` / `status_id_plateau_prob` | `(T,)` | int64 / float32 | Plateau variant of `status_id` |
| `task_progress` | `(T,)` | float32, `0–100` | Smoothstep-interpolated progress within the active step/piece, between each entry's `start_progress`/`end_progress`; holds the last value flat through gaps and across pieces of the same step |
| `step_id_vector` / `status_id_vector` | `(T, 7)` | float32 | Full per-class asymmetric-peak score matrix (one column per active `step_id` 0–6), instead of the collapsed argmax + score pair above |
| `step_id_plateau_vector` / `status_id_plateau_vector` | `(T, 7)` | float32 | Plateau variant of the vectors above |
| `task_progress_vector` | `(T, 7)` | float32 | Per-class progress matrix (progress accumulated per step_id column, 0 elsewhere) |

`debug_frames` (not saved to the `.pt`, only used for
`--save-plots`/label-debug plotting) exposes the same five raw score
matrices before they're collapsed to a single winner per frame.
