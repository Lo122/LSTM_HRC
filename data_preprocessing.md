## Feature Extraction

Each extracted feature file is saved as `features__cam-XX_uid-YY_take-ZZ.pt`.

The filename is derived from the corresponding keypoint file:

`keypoints__cam-XX_uid-YY_take-ZZ.pt` -> `features__cam-XX_uid-YY_take-ZZ.pt`

Each `.pt` file is a Python dictionary saved with PyTorch. The source pose tensor is read from `smoothed_landmarks` with shape `(T, 17, 2)`, where `T` is the number of frames and `17` follows the COCO keypoint order from [data_proc_2d/src/yolo_pose_config.py](data_proc_2d/src/yolo_pose_config.py).

```python
feature_file = {
    "metadata": {...},
    "features": {
        "velocity_scale": ...,
        "acceleration_scale": ...,
        "velocity_xy": ...,
        "acceleration_xy": ...,
        "pol_vectors": ...,
        "pol_distance": ...,
        "pol_angles": ...,
        "pol_distance_velocity": ...,
        "pol_angular_velocity": ...,
        "joint_angles": ...,
        "ratios": ...,
        "dist_ratios": ...,
    },
    "labels": {
        "step_id": ...,
        "step_id_prob": ...,
        "status_id": ...,
        "status_id_prob": ...,
        "task_progress": ...,
        "step_id_plateau": ...,
        "step_id_plateau_prob": ...,
        "status_id_plateau": ...,
        "status_id_plateau_prob": ...,
        "step_id_vector": ...,
        "status_id_vector": ...,
        "step_id_plateau_vector": ...,
        "status_id_plateau_vector": ...,
        "task_progress_vector": ...,
    },
}
```

### Metadata

`metadata` is copied from the source keypoint file and then extended during feature export.

Common fields are:

| Key | Type | Description |
| --- | --- | --- |
| `video` | str | Source video path used to create or re-filter the pose file. |
| `fps` | float | Frames per second used for frame-to-time conversion. |
| `total_frames` | int | Total number of processed frames. |
| `video_frame_width` | int | Source video width in pixels. |
| `video_frame_height` | int | Source video height in pixels. |
| `raw_landmarks_normalized` | bool | Whether the raw landmarks were already normalized when the pose file was built. |
| `filter_config` | dict | Pose smoothing settings such as `window_size`, `poly_order`, `max_jump`, and `max_hold_frames`. |
| `model` | str | YOLO pose model path when the pose file was extracted directly from video. |
| `device` | str | Device used during pose extraction, for example `cpu` or `cuda`. |
| `tracking` | bool | Whether tracking was enabled during pose extraction. |
| `refilter_source_pt` | str | Present when the pose file was re-filtered from an existing PT file instead of from raw video. |
| `refilter_device` | str | Device used for the re-filtering run. |
| `processing_source` | str | Present for re-filtered pose files, typically `pt`. |
| `label_config` | dict | Label export settings added by `extract_features.py`. In the current script this is `{"buffer": 100.0, "smooth_type": "plateau", "function_type": "bezier"}`. |

Notes:

- `label_config` stores the base configuration object recorded in the export script.
- The exported label tensors still include both `asymmetric_peak` and `plateau` variants, even though the saved `label_config.smooth_type` field is `plateau` by default.

### Feature Tensors

The current export excludes COCO keypoints `13-16` (`left_knee`, `right_knee`, `left_ankle`, `right_ankle`) before building the node-wise feature tensors. That leaves 13 active keypoints:

- `nose`
- `left_eye`
- `right_eye`
- `left_ear`
- `right_ear`
- `left_shoulder`
- `right_shoulder`
- `left_elbow`
- `right_elbow`
- `left_wrist`
- `right_wrist`
- `left_hip`
- `right_hip`

the current feature tensors have these shapes:

| Key | Shape | Description |
| --- | --- | --- |
| `velocity_scale` | `(T, 13)` | Per-keypoint velocity magnitude from `torch.diff(landmarks, dim=0)`, sliced to the 13 retained keypoints. The missing leading frame is padded with the first detected velocity. |
| `acceleration_scale` | `(T, 13)` | Per-keypoint acceleration magnitude from `torch.diff(velocity, dim=0)`, sliced to the 13 retained keypoints. The missing leading frames are padded with the first detected acceleration. |
| `velocity_xy` | `(T, 26)` | XY velocity vectors for the retained keypoints, flattened from `(T, 13, 2)`. |
| `acceleration_xy` | `(T, 26)` | XY acceleration vectors for the retained keypoints, flattened from `(T, 13, 2)`. |
| `pol_vectors` | `(T, 26)` | Polar-coordinate XY offsets from the body center, flattened from `(T, 13, 2)`. |
| `pol_distance` | `(T, 13)` | Distance of each retained keypoint from the body center. |
| `pol_angles` | `(T, 13)` | Polar angle of each retained keypoint around the body center in degrees. |
| `pol_distance_velocity` | `(T, 13)` | Temporal derivative of `pol_distance`, padded back to `T` frames. |
| `pol_angular_velocity` | `(T, 13)` | Temporal derivative of `pol_angles` in degrees, padded back to `T` frames. The saved key keeps the current spelling from code: `angular`. |
| `joint_angles` | `(T, 7)` | Joint-angle features in degrees from `JOINT_ANGLE_TRIPLETS` and `JOINT_ANGLE_TRIPLETS_CAL` in `yolo_pose_config.py`. |
| `ratios` | `(T, 2)` | Ratios between configured inter-keypoint distances from `RATIO_BETWEEN_DISTS`. |
| `dist_ratios` | `(T, 13)` | Distance of each retained keypoint from the body center, normalized by a hip-based baseline distance. |

### Feature definitions

- Body center for `pol_vectors`, `pol_distance`, `pol_angles`, `pol_distance_velocity`, `pol_angular_velocity`, and `dist_ratios`: average of `left_hip`, `right_hip`, `left_shoulder`, and `right_shoulder`.
- `pol_vectors` stores the XY offset of each retained keypoint from that center.
- `pol_distance` is the Euclidean norm of each polar offset.
- `pol_angles` is `atan2(y, x)` of each polar offset in degrees.
- `pol_distance_velocity` is `torch.diff(pol_distance, dim=0)` padded back to length `T`.
- `pol_angular_velocity` is `torch.diff(pol_angles, dim=0)` in degrees, padded back to length `T`.
- Baseline for `dist_ratios`: distance between the body center and the midpoint of `left_hip` and `right_hip`.
- Joint-angle column order from `yolo_pose_config.py`: `left_elbow`, `right_elbow`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`, `neck`.
- Ratio column order from `yolo_pose_config.py`: `elbow/shoulder`, `wrist/shoulder`.

### Label Tensors

The exported label tensors are stored under the `labels` key. In the current `cam-04` export they use 7 retained annotation IDs because `EXCLUDE_STEPS = [7, 8, 9, 10, 11]` removes the `*-BL` labels from `ANNOTATION_CONFIG`.

Retained label IDs are:

| ID | Label |
| --- | --- |
| `0` | `Place Spacer` |
| `1` | `Move Spacer` |
| `2` | `Remove Spacer` |
| `3` | `Align the Piece` |
| `4` | `Place the Piece` |
| `5` | `Screw` |
| `6` | `Mistake` |

the current exported label tensors have these shapes:

| Key | Shape | Description |
| --- | --- | --- |
| `step_id` | `(T,)` | Winning step label ID from the asymmetric-peak step score matrix. |
| `step_id_prob` | `(T,)` | Maximum score selected from `step_id_vector`. |
| `status_id` | `(T,)` | Winning status label ID from the asymmetric-peak status score matrix. |
| `status_id_prob` | `(T,)` | Maximum score selected from `status_id_vector`. |
| `task_progress` | `(T,)` | Maximum task progress value across retained status labels. Values are stored on a `0-100` percentage scale. |
| `step_id_plateau` | `(T,)` | Winning step label ID from the plateau step score matrix. |
| `step_id_plateau_prob` | `(T,)` | Maximum score selected from `step_id_plateau_vector`. |
| `status_id_plateau` | `(T,)` | Winning status label ID from the plateau status score matrix. |
| `status_id_plateau_prob` | `(T,)` | Maximum score selected from `status_id_plateau_vector`. |  
| `step_id_vector` | `(T, 7)` | Per-label asymmetric-peak step scores for the retained annotation IDs. |
| `status_id_vector` | `(T, 7)` | Per-label asymmetric-peak status scores for the retained annotation IDs. |
| `step_id_plateau_vector` | `(T, 7)` | Per-label plateau step scores for the retained annotation IDs. |
| `status_id_plateau_vector` | `(T, 7)` | Per-label plateau status scores for the retained annotation IDs. |
| `task_progress_vector` | `(T, 7)` | Per-label progress curves for the retained annotation IDs on a `0-100` scale. |

Notes:

- `step_id_vector`, `status_id_vector`, `step_id_plateau_vector`, and `status_id_plateau_vector` are clipped to a maximum value of `1.0` after summing overlapping label segments.
- The scalar `*_id` tensors are produced with `idxmax` over the corresponding score matrices.
- The scalar `*_prob` tensors are produced with `max` over the corresponding score matrices.

### Label Curve Generation

Step and status label curves are generated from the annotated frame ranges using `define_step_label_entry(...)` in `data_proc_2d/src/label_definition.py`.

Current configuration in `extract_features.py`:

| Setting | Current value | Effect |
| --- | --- | --- |
| `buffer` | `100.0` | Controls how many frames the smoothing curve extends around the annotation boundary. |
| `function_type` | `bezier` | Uses a bounded cubic easing curve that returns exactly `0.0` outside the active support region. |
| `smooth_type` | `asymmetric_peak` and `plateau` | Both variants are exported separately. |

The two label variants behave as follows:

- `asymmetric_peak`: the label reaches `1.0` at `start_frame`, ramps in over `buffer` frames before the start, and then decays toward zero up to approximately `end_frame + buffer`.
- `plateau`: the label stays at `1.0` from `start_frame` through `end_frame`, with symmetric falloff controlled by `buffer` on both sides.

Task progress curves are generated differently:

- `start_progress` and `end_progress` are first assigned per `(step_id, piece_id)` group in the processed annotation data.
- Inside an annotated status segment, progress is linearly interpolated from `start_progress` to `end_progress`.
- If the same `(step_id, piece_id)` appears again later, the gap between the two segments is filled with the last completed progress value instead of resetting to zero.
- Once a segment reaches `100`, the next group instance starts over from `0`.
