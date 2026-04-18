
# Folder and File Naming Guide

This document summarizes the recommended naming rules and folder structure for raw videos, processed datasets, and the GitHub repository.

## 1. File Naming Convention

### Base pattern

`{prefix}__cam-{camera_id}_uid-{user_id}_take-{take_id}.{ext}`

### Field definitions

- `prefix`: file type such as `video`, `label`, `raw`, or `norm`
- `camera_id`: two-digit camera ID, for example `01`
- `user_id`: two-digit participant ID from the participant spreadsheet
- `take_id`: two-digit take index for that camera

### Example

`video__cam-01_uid-03_take-02.mp4`

### Recommended rules

- Use zero-padded IDs such as `01`, `02`, and `03` so files sort correctly.
- Keep the token order fixed as `cam`, `uid`, then `take`.
- Reuse the same metadata pattern for all derived files.

Examples:

- `video__cam-01_uid-03_take-02.mp4`
- `label__cam-01_uid-03_take-02.csv`
- `label__cam-01_uid-03_take-02.json`
- `raw__cam-01_uid-03_take-02.pt`
- `norm__cam-01_uid-03_take-02.pt`

## 2. Google Drive Structure

Recommended structure:

```text
videos/
    raw/
        cam-01/
            video__cam-01_uid-03_take-02.mp4
    cropped/
        cam-01/
            video__cam-01_uid-03_take-02.mp4
    annotations/
        cam-01/
            label__cam-01_uid-03_take-02.csv
    dataset/
        annotations/
            cam-01/
                label__cam-01_uid-03_take-02.json
        train/
            raw/
                raw__cam-01_uid-03_take-02.pt
            norm/
                norm__cam-01_uid-03_take-02.pt
        test/
            raw/
                raw__cam-01_uid-03_take-02.pt
            norm/
                norm__cam-01_uid-03_take-02.pt
```

Notes:

- `raw/`: original camera recordings
- `cropped/`: clips containing a single task iteration
- `annotations/`: ELAN-exported CSV labels
- `dataset/`: tensors and annotation files used for training and evaluation

## 3. GitHub Repository Structure

Current top-level structure in this repository:

```text
application/        # application entry points and viewer-related code
data_proc_2d/       # 2D data preprocessing pipeline
data_proc_3d/       # 3D data preprocessing pipeline
design/             # design notes and supporting documents (rhino gh files, etc.)
logs/               # runtime and experiment logs
model_data/         # pretrained weights and checkpoints
model_lstm/         # LSTM model training code and experiments
model_rnn/          # RNN baseline code
model_transformer/  # transformer model code
post_proc/          # post-processing scripts and outputs
reference/          # notebooks and external reference material
robot_control/      # robot-side control logic
utilities/          # shared utility code
```

second level structure:

```text
app/                # application entry points and scripts
dataset/            # processed datasets used for training and evaluation
results/            # experiment results and outputs
src/                # source code for each module
```


## 4. Suggested Naming Improvements

| Current name | Suggested name | Reason |
| --- | --- | --- |
| `videos/dataset` | `videos/processed` | These files are derived artifacts rather than raw recordings. |
| `label_info` | `annotations` | More standard and more descriptive. |
| `model_lstm` | `models/lstm` | Makes model families easier to organize. |
| `model_rnn` | `models/rnn` | Keeps naming consistent across architectures. |
| `model_transformer` | `models/transformer` | Keeps naming consistent across architectures. |
| `model_data` | `weights` or `checkpoints` | Clearer if the folder mainly stores trained model files. |