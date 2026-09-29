# Claude L1 LSTM alignment — current state

Updated: 2026-09-29. The latest user request authorizes full alignment with the supplied
`LSTM_GRU_Model_Settings.md`, using **L1_lstm**, not GRU S3. This section supersedes the
historical staged plan below. Baseline files and historical runs are preserved.

## Current settings

| Item | Aligned value |
| --- | --- |
| Backbone | One-layer causal LSTM, hidden 128; shared Linear/ReLU/Dropout 0.3 |
| Input | 16 panels in Claude order, 251 columns |
| Feature transforms | Azimuth degrees -> sin/cos (all sine columns then all cosine columns); ratios clipped [0,4], before normalization |
| Windows | 120 consecutive samples at 30 fps; hop 10 raw frames; target is last frame |
| Invalid frames | Preserve timeline; reject each window containing any non-finite source feature frame |
| Optimizer | AdamW, lr 1e-3, weight_decay 1e-4 |
| Training | Batch 256, 12 epochs, uniform shuffle, no replacement |
| LR / gradients | CosineAnnealingLR T_max=12, step once after each epoch; clip global gradient norm at 1 |
| Seed | torch.manual_seed(fold_index) immediately before model construction; explicit shuffle generator with the same seed |
| Heads | 7 task logits, 7 progress values, 1 mistake logit, 1 idle logit (existing four-head design retained) |
| Loss | Existing peak BCE with task pos_weight cap 2; active-lane progress MSE; binary mistake/idle BCE |
| Loss weights | task 1, progress 0.3, mistake 0.3, idle 0.2 |
| Checkpoints | Retain both minimum total validation loss and maximum validation task macro-F1; evaluate the latter |
| Split | 15-fold LOSO; two validation subjects selected by RandomState(1000 + test_uid) |
| Augmentation | Original + exactly one _aug-01 mirror per training recording; original-only val/test; no _aug-02/03 |
| Normalization | Per fold, only training originals/mirrors; sample every seventh original frame, then filter non-finite rows |
| Task evaluation | Seven-class macro-F1 on true non-idle windows; truth argmax(plateau + 0.001 * peak) |
| Additional evaluation | Eight-class end-to-end macro-F1/accuracy using predicted idle > 0; supplementary, not used for L1 checkpoint selection |

## Files and execution

- `LSTM_train_mirror_tune.ipynb`: existing notebook, updated in place; defaults to test UID 2,
  val UID 5/14. Stale outputs were cleared because they describe a different experiment.
- `LSTM_model_train_tune.py`: preserves the four heads/loss; aligns default hidden/input/dropout.
- `LSTM_databuilder_tune.py`: validates normalization identity and rejects invalid windows without
  joining frames across gaps. Counts class weights on surviving target indices.
- `../data_proc_2d/app/build_norm_dataset_tune.py`: exports `claude_l1/fold_XX/` containing
  `norm_stats.npz`, a dataset manifest, and normalized train/val/test NPZ files. Sources are never moved.
  Existing `norm/`, `norm_tune/` and dated statistics files are not overwritten by the new main entry.
- `LSTM_engine_tune.py`: shared epoch and checkpoint loop for notebook and LOSO; avoids two training implementations.
- `LSTM_loso_tune.py`: prepares and trains all 15 folds, then writes per-subject mean/std metrics.
- `test_LSTM_tune.py`: semantic, preprocessing, checkpoint and notebook/runner integration checks.

For **one fold**, set matching `data_root` and `test_uid` in the exporter and notebook. First run
`python -B data_proc_2d/app/build_norm_dataset_tune.py`, then run the notebook from the beginning.
The notebook needs the new fold export; old `norm_tune` NPZs are not interchangeable with it.

For **all 15 folds**, set `data_root` in `LSTM_loso_tune.py`, then run from the repository root:

```powershell
& '..\hrc_communication\.venv\Scripts\python.exe' -B -m model_lstm.LSTM_loso_tune
```

This full runner performs preprocessing and training sequentially for each fold. It saves into
`model_lstm/runs_3d_tune/L1_LOSO_<timestamp>/`, with separate checkpoints and predictions per fold.
Fold exports and their memory-mapped caches are retained for reuse and therefore require disk space.
No full preprocessing or training was started during this code change.

Each NPZ records its feature profile and statistics identity. The notebook checks the statistics
file hash, and its loader rejects NPZs from another normalization export. Each run also saves a
copy of its statistics and data manifest. Cached fold exports are reused only if source paths,
sizes, timestamps and preprocessing settings still match; changed sources require a new export root.

## Verified and remaining limits

- All **18 tests passed**, including an end-to-end notebook run and a two-epoch standalone fold run.
- Tested actual cosine LR changes, gradient-clipping calls, both checkpoint selection rules,
  finite-window endpoint counts, train-only statistics, 251-dimensional transforms, absence of
  raw-PT dependency after export, and subject-wise LOSO aggregation.
- Read-only checks resolved the source/mirror manifests for all 15 real folds. Test UID 2 selects
  144 training files (72 originals + mirrors), 12 validation originals, and 9 test originals.
- A real original recording yielded 251 transformed dimensions and matching eight-lane targets.
- This is document-level protocol alignment, not a claim of reproducing macro-F1 0.456.
  Claude's source code, exact data snapshot and fold results remain unavailable. Existing target
  vectors are preserved, not regenerated from an assumed new annotation scheme.
- Explicit choices for details unspecified in the document: float64 statistics accumulation,
  population std (ddof=0), sampling original indices before finite-row filtering, deterministic
  sorted file order, all seven labels in macro-F1, fold std ddof=0, and an independent shuffle generator.
- Progress outputs remain linear. Missing positive mistake training windows raise an error;
  no undocumented fallback weighting is applied.

---

# Historical staged experiment record (superseded where different)

# LSTM tuning decisions and experiment record

Updated: 2026-09-29. This file records the user's agreed direction for future sessions.

## Agreed direction

- Keep LSTM. Use Claude's `L1_lstm` settings as the reference, not the GRU S3 model.
- Make changes in stages, so their effects can be compared.
- Create new files by appending `_tune` to the original names. Preserve the original implementations.
- Start with **stage 3: labels, output heads and loss**. Hyperparameter/input/split changes come later.
- Follow-up on 2026-09-29: replace `WeightedRandomSampler` with ordinary training shuffle, keeping the current loss settings.
- Claude's original code/results are unavailable; only `LSTM_GRU_Model_Settings.md` was provided.

## Current experiment: stage 3

Files:

- [build_norm_dataset_tune.py](../data_proc_2d/app/build_norm_dataset_tune.py): keep existing splits or use the original UID split algorithm; compute training-only statistics and export normalized features plus complete labels to each split's `norm_tune/`.
- `LSTM_databuilder_tune.py`: read both features and targets from `norm_tune/*.npz`; retain memory-mapped caches from the mirror notebook. Raw PT files are needed only during preprocessing.
- `LSTM_model_train_tune.py`: same LSTM/shared-layer structure, with task 7 / progress 7 / mistake 1 / idle 1 outputs and `AssistLoss`.
- `LSTM_train_mirror_tune.ipynb`: training, checkpoint selection, metrics, plots and per-lane CSV exports for these outputs.
- `test_LSTM_tune.py`: semantic tests and an end-to-end synthetic notebook experiment.

The pre-existing `LSTM_train_tune.ipynb` is unrelated and is not overwritten.

Labels and loss:

| Head | Target | Loss | Coefficient |
| --- | --- | --- | ---: |
| task | `task_id_vector[:7]` (soft peak) | BCEWithLogits; per-lane neg/pos capped at 2 | 1.0 |
| progress | `task_progress_vector[:7]` (already scaled to [0,1] in NPZ) | MSE on `plateau[:7] >= 0.5`, divided by active-lane count | 0.3 |
| mistake | binary `mistake` | BCEWithLogits; uncapped neg/pos | 0.3 |
| idle | `task_id_plateau_vector[7] >= 0.5` | BCEWithLogits | 0.2 |

Task BCE includes idle windows and preserves any nonzero soft peak targets. Empty progress masks produce differentiable zero. All pos_weights are counted once from training window endpoints. A task with no positive endpoints gets the cap of 2; no positive mistake endpoints raises an explicit error because the uncapped ratio is undefined.

Task metrics use `argmax(plateau + 0.001 * peak)` on true non-idle windows, with all seven task labels included in macro-F1. Progress MAE uses active lanes of true non-idle windows, in percentage points. Idle/mistake metrics cover all windows. Progress outputs are raw linear values; no added sigmoid or clipping. Evaluation restores the best validation task-F1 checkpoint.

The original 91 features, file sets/augmentation, existing participant split, 160-frame window / hop 30, hidden 64, dropout 0.2, one layer, Adam, lr 1e-4, weight decay 0.01 and batch 32 remain the defaults. The notebook is currently configured for 30 epochs. Training uses `DataLoader(shuffle=True)` with seed 42 and no replacement: every window is used once per epoch. Validation/test and training evaluation retain `shuffle=False`. Scalar `task_id` counts are only used for distribution plots and metadata. BCE pos_weights remain unchanged. Normalization uses the original formula, with statistics recomputed from the configured training folder. Old CE weights and the class-7 loss multiplier are replaced by the new losses. Other hyperparameters and the evaluation protocol still differ from Claude L1.

Each NPZ must contain the normalized features plus `task_id`, `mistake`, `task_progress`, `task_id_vector`, `task_id_plateau_vector` and `task_progress_vector`. The three vectors retain all eight lanes; peak/plateau are unchanged, and scalar/vector progress are divided by 100 exactly once by the exporter. Idle is still derived from plateau lane 7. Old NPZ files lacking vectors produce a clear regeneration error. The Dataset no longer accepts `raw_pt_path` or loads PT files. The recorded normalization-file hash is provenance metadata, not a new verification of every normalized value.

Loader simplification: cache cleanup uses context managers while preserving atomic publication and memory mapping. The NPZ-only cache uses a new version key to avoid mixing old PT label caches with scaled NPZ labels. Labels share one float32 conversion, and window target slices are stored once. The loader retains field/shape/window checks but no longer scans every feature and label for finite values and numeric ranges at initialization; it expects the preprocessed dataset's documented ranges. Training still rejects non-finite loss.

After simplification, all 9 tests passed, including interrupted-cache retry and the notebook's one-epoch end-to-end test.

After the NPZ-only migration, all 11 tests passed, including shared-statistics export for all three splits, one-time progress scaling, missing-vector errors, and a complete notebook run with all raw PT fixtures removed. A temporary export of real training recording `features__cam-05_uid-03_take-01_aug-01.pt` reproduced all 16 normalized feature panels exactly versus the existing NPZ and loaded as 391 windows of `(160, 91)` selected inputs. No bulk data export or full training was run during implementation.

Outputs go to `model_lstm/runs_3d_tune/exp_<timestamp>/`. Existing notebook outputs are from earlier runs; rerun cells in order for the updated sampling behavior. The training cell starts the full configured experiment. Current cache lives under the already-ignored `model_lstm/.dataset_cache/tune/`.

Before running the notebook, export all three splits from the repository root:

```powershell
& '..\hrc_communication\.venv\Scripts\python.exe' -B data_proc_2d/app/build_norm_dataset_tune.py
```

Edit `seg_pt_train_dir`, `seg_pt_val_dir`, and `seg_pt_test_dir` in the exporter's `__main__` block to point to the three raw PT folders. There are no command-line parsers. If all three contain PT files, splitting is skipped, even if `original_pt_dir` is unavailable. If all are empty, files from `original_pt_dir` are moved into these folders using the original `split_by_uid` algorithm (approximately 80/10/10, seed 42, 5000 trials). A partial split raises an error before moving files.

Each run computes mean/std from all training frames for every panel in `FEATURE_KEYS`, using the original `mean(axis=0)` and `std(axis=0) + 1e-6` formula. `np.savez` writes `data_proc_3d/dataset/norm_mirrored_<YYYY-MM-DD>.npz` with the run's local date. It contains `<feature>_mean` and `<feature>_std` arrays for later normalization. All three splits use these same training statistics; val/test never contribute to fitting them. The exported feature/label files remain compressed NPZ under each raw folder's sibling `norm_tune/`, preserving raw contents and old `norm/` outputs. Re-running on the same day replaces the statistics file and matching exports. Update the notebook's data/statistics paths to match the export; the notebook still selects the original seven feature panels.

Verified after this entry-point change: all 14 tests passed, including exact agreement with the original UID splitting algorithm, skipping an existing split, rejecting a partial split, and fitting statistics only on train. The main block was also executed with temporary paths and synthetic data: it produced `norm_mirrored_2026-09-28.npz` through `np.savez` and normalized all three splits correctly. Real dataset folders were not processed or moved.

Verified on 2026-09-29 after switching to shuffle: all 14 tests passed. The notebook smoke test checks that each window appears exactly once in each of two sampler iterations, runs one training epoch, and verifies the saved shuffle metadata. Existing notebook outputs were preserved; full training was not rerun.

## Remaining stages

| Area | Current default | Claude L1 reference |
| --- | --- | --- |
| hidden / shared dropout | 64 / 0.2 | 128 / 0.3 |
| optimizer / lr / weight decay | Adam / 1e-4 / 0.01 | AdamW / 1e-3 / 1e-4 |
| batch / epochs | 32 / 30 | 256 / 12 |
| scheduler / gradient clipping | none / none | cosine T_max=12 / norm 1 |
| window / hop | 160 / 30 | 120 / 10, continuous 30 fps |
| input | 91 dimensions | 251 dimensions, all panels, azimuth sin/cos, ratios clipped [0,4] |
| normalization | all-frame statistics from configured train folder, shared by train/val/test | training-only statistics per fold, every seventh frame |
| sampling | uniform shuffle (aligned on 2026-09-29) | uniform shuffle |
| augmentation | existing augmented file sets | originals + one mirror per training recording; original val/test |
| evaluation split | fixed existing participants | 15-fold LOSO, with two validation participants per fold |

The order after stage 3 is not yet fixed. Do not apply these changes automatically in a future edit without considering the next requested experiment. Target L1's reported 0.456 ± 0.072 macro-F1 only once the whole evaluation protocol is aligned; fixed-split scores are not directly comparable.

## Verification

Run from the repository root in the installed PyTorch environment:

```powershell
& '..\hrc_communication\.venv\Scripts\python.exe' -B -m unittest model_lstm.test_LSTM_tune -v
```

The synthetic test executes the notebook from loading through one training epoch, checkpoint reload, evaluation, plots and CSV export, with temporary data/results. Full training is a separate user-run experiment; a smoke test cannot establish performance improvement.

Verified on 2026-09-28: all 8 tests passed. One paired file from each real split also loaded successfully (train UID 03: 391 windows; val UID 01: 581; test UID 02: 278). Each checked first input window was exactly equal to the original normalized NPZ features, with shape `(160, 91)`. A real CUDA batch of 32 windows completed forward/backward and an optimizer step with finite loss and gradients; this smoke check used unit pos_weights, while the notebook computes weights from all training windows. No full training run or performance comparison has been performed.
