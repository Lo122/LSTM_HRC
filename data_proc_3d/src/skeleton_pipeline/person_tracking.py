"""Whole-video person tracking + target selection for OFFLINE data generation.

person_selection.select_tracked_person_keypoints() picks, frame by frame,
whichever detection best matches the previous pose. That is the right shape
of decision for a LIVE pipeline (it is causal -- it only ever looks at the
past), but it is a weak one: pose similarity is a poor identity cue (two
people standing similarly look alike), and a bystander who walks between
the camera and the subject can capture the track. Once captured, the
"previous pose" reference is the wrong person, so the error is sticky.

Offline we are not bound by that. The whole video is available, so the
target can be chosen GLOBALLY instead of frame by frame:

  pass 1  run YOLO with a real multi-object tracker (BoT-SORT/ByteTrack --
          Kalman motion prediction, occlusion handling, optional appearance
          ReID) and collect every track ID over the whole clip.
  select  the subject performing the task is present for most of the video;
          people who merely walk through are transient. So the longest-lived
          track is the target.
  pass 2  emit only that track's keypoints (see the caller).

This costs one extra video decode but NOT an extra YOLO pass -- pass 1
caches every track's keypoints, so the lifting pass just looks them up.

Note this does not weaken the train/deployment matching argument the
generate_lstm_training_data docstring makes: WHICH human the pipeline
points at is a data-cleaning concern, not part of the modelled signal.
Everything downstream of the target choice (the lifter, its window, the
filters, the features) is untouched and still matches deployment.
"""
import json
from pathlib import Path

import numpy as np

# Same body-scale/center definitions the frame-by-frame selector uses, so
# "one body width" means the same thing in both places.
from skeleton_pipeline.person_selection import _tracking_center, _tracking_scale


def collect_person_tracks(yolo, cap, imgsz=None, max_frames=None,
                          tracker="botsort.yaml", device=None, progress_every=500):
    """Pass 1: run the tracker over every frame, keeping each track's keypoints.

    Returns ``(tracks, n_frames)`` where ``tracks`` is
    ``{track_id: {frame_idx: (keypoints_xy (17,2), keypoints_conf (17,))}}``.

    ``cap`` is consumed (read to the end / to ``max_frames``); the caller is
    expected to reopen it for the lifting pass.
    """
    kwargs = {}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if device is not None:
        kwargs["device"] = device

    tracks = {}
    frame_idx = 0
    while max_frames is None or frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # persist=True keeps tracker state across calls for this stream.
        results = yolo.track(frame, persist=True, tracker=tracker, verbose=False, **kwargs)
        if results:
            frame_result = results[0]
            boxes = frame_result.boxes
            keypoints = frame_result.keypoints
            # boxes.id is None when the tracker assigned no ID this frame
            # (e.g. detection below track_high_thresh) -- skip those.
            if keypoints is not None and boxes is not None and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
                all_xy = keypoints.xy.cpu().numpy()
                all_conf = (keypoints.conf.cpu().numpy()
                            if keypoints.conf is not None else None)
                for i, track_id in enumerate(track_ids):
                    conf = (all_conf[i] if all_conf is not None
                            else np.ones(all_xy.shape[1], dtype=np.float32))
                    tracks.setdefault(int(track_id), {})[frame_idx] = (all_xy[i], conf)

        frame_idx += 1
        if progress_every and frame_idx % progress_every == 0:
            print(f"    [track] frame {frame_idx}, {len(tracks)} track(s) so far")

    return tracks, frame_idx


def _usable_frame_count(frames, conf_threshold, min_valid_joints):
    count = 0
    for _xy, conf in frames.values():
        if int(np.sum(np.asarray(conf) >= conf_threshold)) >= min_valid_joints:
            count += 1
    return count


def select_dominant_track_id(tracks, conf_threshold=0.2, min_valid_joints=4):
    """Pick the track that is usable for the most frames -- the task subject.

    Frames are only counted when the detection actually carries enough
    confident joints, so a bystander who is half out of frame for a long
    time cannot out-count the real subject on raw appearance count alone.

    Returns ``(track_id, usable_frame_count)``, or ``(None, 0)`` if nothing
    qualifies.
    """
    best_id, best_count = None, 0
    for track_id, frames in tracks.items():
        count = _usable_frame_count(frames, conf_threshold, min_valid_joints)
        if count > best_count:
            best_id, best_count = track_id, count
    return best_id, best_count


def _usable_frames_sorted(frames, conf_threshold, min_valid_joints):
    return sorted(
        idx for idx, (_xy, conf) in frames.items()
        if int(np.sum(np.asarray(conf) >= conf_threshold)) >= min_valid_joints
    )


def _robust_scale(frames, indices, conf_threshold):
    """Median body scale over several frames.

    A single frame's scale is the extent of its CONFIDENT keypoints, so a
    partial detection (head and shoulders only) reports a much smaller body
    than the same person fully detected. The frames either side of a track
    break are exactly where partial detections cluster -- the track broke
    because detection degraded -- so measuring scale on the boundary frame
    alone is badly biased: on cam-05_uid-04_take-02 it read 220px at the end
    of one fragment and 475px at the start of the next, a spurious 2.16x
    "size change" for a person who had not moved (their centers were 95px
    apart). Taking the median over a handful of frames removes that bias.
    """
    scales = [_tracking_scale(*frames[i], conf_threshold) for i in indices]
    scales = [s for s in scales if s > 1e-6]
    return float(np.median(scales)) if scales else 0.0


def _gap_is_unoccupied(spans, gap_start, gap_end, ignore_ids):
    """True when no substantial tracklet covers any of the open gap.

    Distinguishes the two reasons the subject's track can end: they walked OUT
    OF FRAME (nobody on screen at all until they return -- whatever appears
    next is them), or they were OCCLUDED while someone else was around (the
    next tracklet could be that other person).

    Only the first is safe to bridge across a long gap, and that distinction
    is what the gap length alone cannot express: on cam-06_uid-06_take-01 a
    1.8s frame-out and on cam-07_uid-10_take-01 a 12s occlusion with a second
    person working are both "too long", but only one of them is ambiguous.
    """
    for track_id, (start, end) in spans.items():
        if track_id in ignore_ids:
            continue
        if start < gap_end and end > gap_start:
            return False
    return True


def _clean_shared_frames(frames_a, frames_b, shared, conf_threshold,
                         reference_a, reference_b, min_scale_fraction=0.5):
    """Shared frames where BOTH detections cover a plausible fraction of the body.

    A tracklet's first and last frames are its worst: the track was born
    because detection had only just become good enough, and it died because
    detection degraded. Those frames routinely carry a partial skeleton -- a
    couple of confident joints -- whose extent is a small fraction of the real
    body, which makes the centre meaningless for comparison.

    Overlaps are made of exactly those frames, so they must be screened. On
    cam-07_uid-10_take-01 the junction at 655s shared only 6 frames, and the
    3 born-frames reported a body scale of ~124px against the track's true
    ~572px; their centres were nonsense, distances 1.42-1.60, and the median
    over all 6 came to 0.937 -- rejected. The 3 valid frames agreed at
    0.39-0.45. Screening the partials leaves a median of 0.41, and the video
    keeps its second half instead of ending at 10.9 of 22.3 minutes.
    """
    clean = []
    for idx in shared:
        scale_a = _tracking_scale(*frames_a[idx], conf_threshold)
        scale_b = _tracking_scale(*frames_b[idx], conf_threshold)
        if scale_a >= min_scale_fraction * reference_a and scale_b >= min_scale_fraction * reference_b:
            clean.append(idx)
    return clean


def _overlap_distance(frames_a, frames_b, shared, conf_threshold):
    """Median centre distance (in body widths) over frames where BOTH
    tracklets have a detection.

    This is the strongest identity evidence available, and it is only
    available when tracklets overlap. Two tracklets that overlap are either
    one person detected twice -- which is what a tracker handoff looks like,
    it spawns the replacement id a few frames before retiring the old one --
    or two people who were genuinely visible at the same moment. Measured
    across cam-05/06/07, the two cases do not come close to touching:
    handoff duplicates sit on top of each other (median 0.00-0.30 body
    widths), while distinct people are at least half a body apart
    (0.65-2.73). Comparing the tracklets' ENDPOINTS instead, as an earlier
    version did, is far weaker -- the last frame before a track dies is its
    worst-detected one.
    """
    distances = []
    for idx in shared:
        center_a = _tracking_center(*frames_a[idx], conf_threshold)
        center_b = _tracking_center(*frames_b[idx], conf_threshold)
        if center_a is None or center_b is None:
            continue
        scale_a = _tracking_scale(*frames_a[idx], conf_threshold)
        scale_b = _tracking_scale(*frames_b[idx], conf_threshold)
        distances.append(
            float(np.linalg.norm(center_b - center_a)) / max(min(scale_a, scale_b), 1.0))
    return float(np.median(distances)) if distances else None


def _junction_continuous(pose_a, pose_b, scale_a, scale_b, conf_threshold,
                         max_center_jump_ratio, min_scale_ratio, max_scale_ratio):
    """Are two tracklets, either side of a break, the same person?

    Deliberately the SAME kind of jump/scale test that proved unreliable
    frame by frame, applied where it IS reliable: both sides are detections
    the tracker already committed to, and the question is asked a couple of
    dozen times per video rather than once per frame.

    POSITION is the primary cue -- it cleanly separates the subject
    (0.43 body-widths across the break) from a bystander who happened to be
    visible nearby (3.37). SCALE is only a loose sanity check, because across
    a gap of up to max_gap_frames a person can genuinely walk toward the
    camera and grow substantially.
    """
    center_a = _tracking_center(*pose_a, conf_threshold)
    center_b = _tracking_center(*pose_b, conf_threshold)
    if center_a is None or center_b is None:
        return False
    if scale_a <= 1e-6 or scale_b <= 1e-6:
        return False

    ratio = scale_b / scale_a
    if not (min_scale_ratio <= ratio <= max_scale_ratio):
        return False

    # Normalize by the SMALLER body, so a jump between people at different
    # depths is not deflated by the nearer (larger) one -- the mistake that
    # made the frame-by-frame guard miss a bystander swap entirely.
    distance = float(np.linalg.norm(center_b - center_a)) / max(min(scale_a, scale_b), 1.0)
    return distance <= max_center_jump_ratio


def link_track_fragments(tracks, target_id, conf_threshold=0.2, min_valid_joints=4,
                         max_gap_frames=45, max_duplicate_distance=0.5, min_fragment_frames=15,
                         max_center_jump_ratio=1.5,
                         min_scale_ratio=0.5, max_scale_ratio=2.0, scale_window=15,
                         min_duplicate_scale_ratio=0.65, max_duplicate_scale_ratio=1.55,
                         max_empty_gap_frames=1800):
    """Stitch the subject's other track IDs back onto the target track.

    A multi-object tracker gives an id per *tracklet*, not per person: a long
    occlusion, a detection dropout, or the subject leaving and re-entering
    ends one id and starts another. Taking only the single longest id
    therefore silently discards real subject data -- on
    cam-05_uid-04_take-02 the subject is id=1 (frames 0..292), id=2
    (290..6407) and id=77 (6416..7270), so keeping only id=2 dropped the
    first 10 s and the last 28 s of the video.

    A fragment must extend the target's range (one that sits entirely inside
    it adds nothing), and then has to pass one of two tests depending on
    whether it OVERLAPS the target in time:

      * overlapping -- the two tracklets share frames, so compare them
        directly on those frames (_overlap_distance). A tracker handoff
        spawns the replacement id a few frames before retiring the old one,
        so the same person appears as two detections sitting on top of each
        other; two genuinely different people are far apart. This is the
        strong test.
      * disjoint -- nothing to compare directly, so fall back to continuity
        across the break (_junction_continuous), bounded by max_gap_frames.

    Note overlap is NOT itself disqualifying. An earlier version rejected any
    overlap beyond a few frames, on the theory that co-existing means "two
    people". Real handoffs overlap by more than that (17 frames on
    cam-07_uid-01_take-02, 25 on cam-06_uid-01_take-02), so that rule threw
    away most of both videos -- cam-07 kept only 41% of its frames, losing
    everything after 243.9s.

    Returns ``(merged_frames, merged_ids)``.
    """
    # Spans of every tracklet big enough to be a person rather than detector
    # noise. Used to tell a frame-out (nobody on screen) from an occlusion
    # with someone else around -- see the gap handling below.
    substantial_spans = {}
    for other_id, other_frames in tracks.items():
        other_usable = _usable_frames_sorted(other_frames, conf_threshold, min_valid_joints)
        if len(other_usable) >= min_fragment_frames:
            substantial_spans[other_id] = (other_usable[0], other_usable[-1])

    merged = dict(tracks[target_id])
    merged_ids = []
    used = {target_id}

    while True:
        target_frames = _usable_frames_sorted(merged, conf_threshold, min_valid_joints)
        if not target_frames:
            break
        cur_start, cur_end = target_frames[0], target_frames[-1]

        best = None  # (gap, track_id, frames)
        for track_id, frames in tracks.items():
            if track_id in used:
                continue
            usable = _usable_frames_sorted(frames, conf_threshold, min_valid_joints)
            if len(usable) < min_fragment_frames:
                continue
            f_start, f_end = usable[0], usable[-1]

            if f_end <= cur_end and f_start >= cur_start:
                continue                            # entirely inside; adds nothing

            shared = sorted(set(target_frames) & set(usable))
            if shared:
                # Screen out partial detections first (see
                # _clean_shared_frames): they cluster at exactly the frames an
                # overlap is made of. If NOTHING survives the screen the
                # overlap carries no usable evidence, so fall through to the
                # continuity test below rather than rejecting -- a 3-frame
                # overlap of two bad frames is not grounds to refuse a merge,
                # and treating it as such cost cam-07_uid-06_take-02 its first
                # 196 seconds.
                reference_target = _robust_scale(merged, target_frames, conf_threshold)
                reference_candidate = _robust_scale(frames, usable, conf_threshold)
                shared = _clean_shared_frames(
                    merged, frames, shared, conf_threshold,
                    reference_target, reference_candidate)

            if shared:
                # Overlapping tracklets: ask the direct question -- are these
                # two detections of the same person at the same moment?
                distance = _overlap_distance(merged, frames, shared, conf_threshold)
                if distance is None or distance > max_duplicate_distance:
                    continue                        # two people, not a handoff

                # Position alone is not enough. Two people at very different
                # DEPTHS can project to nearby image positions -- a bystander
                # standing further down the room lines up behind the subject.
                # At the same instant one person cannot have two body sizes,
                # so the scale ratio separates them where position cannot:
                # on cam-07_uid-01_take-02 at ~310s the subject goes behind a
                # column and a bystander by the door (scale 266px vs the
                # subject's 656px, ratio 0.39) sat only 0.41 body widths away
                # in image space and was wrongly merged. Real handoffs of one
                # person measure 0.80-1.35 here.
                scale_target = _robust_scale(merged, shared, conf_threshold)
                scale_candidate = _robust_scale(frames, shared, conf_threshold)
                if scale_target <= 1e-6 or scale_candidate <= 1e-6:
                    continue
                scale_ratio = scale_candidate / scale_target
                if not (min_duplicate_scale_ratio <= scale_ratio <= max_duplicate_scale_ratio):
                    continue                        # same place, different size -> different person
                gap = 0
            else:
                # Disjoint tracklets: no shared frame to compare, so fall
                # back to continuity across the break.
                if f_start > cur_end:
                    gap = f_start - cur_end
                    gap_start, gap_end = cur_end, f_start
                    junction = (merged[cur_end], frames[f_start])
                    scales = (_robust_scale(merged, target_frames[-scale_window:], conf_threshold),
                              _robust_scale(frames, usable[:scale_window], conf_threshold))
                else:
                    gap = cur_start - f_end
                    gap_start, gap_end = f_end, cur_start
                    junction = (frames[f_end], merged[cur_start])
                    scales = (_robust_scale(frames, usable[-scale_window:], conf_threshold),
                              _robust_scale(merged, target_frames[:scale_window], conf_threshold))

                # A long gap is only ambiguous if somebody else was on screen
                # during it. When the frame is empty the subject simply walked
                # out, and whoever walks back in is them -- so allow a much
                # longer gap in that case. Frame-outs routinely exceed
                # max_gap_frames (1.8s and 4.0s on the uid-06 takes, against a
                # 1.5s limit) and truncating there loses the whole remainder of
                # the video.
                limit = max_gap_frames
                if gap > limit and _gap_is_unoccupied(
                        substantial_spans, gap_start, gap_end, {target_id, track_id, *used}):
                    limit = max_empty_gap_frames
                if gap > limit:
                    continue
                if not _junction_continuous(*junction, *scales, conf_threshold,
                                            max_center_jump_ratio,
                                            min_scale_ratio, max_scale_ratio):
                    continue
            if best is None or gap < best[0]:
                best = (gap, track_id, frames)

        if best is None:
            break

        _gap, track_id, frames = best
        for idx, pose in frames.items():
            merged.setdefault(idx, pose)   # never overwrite the target's own frames
        used.add(track_id)
        merged_ids.append(track_id)

    return merged, merged_ids


def load_track_overrides(path):
    """Load a hand-written {video_stem: [track_id, ...]} JSON override map.

    Automatic target selection is a heuristic, and some clips are genuinely
    ambiguous -- on cam-07_uid-10_take-01 the subject stops working, watches a
    second person perform the task, then leaves the frame entirely, so no
    geometric rule can decide which tracklets are "the subject" without
    re-identifying her across a 90s absence. This is OFFLINE training-data
    generation, so the cheap and reliable answer is to let a human name the
    track ids for such clips and skip the guessing.

    Accepted per-video forms (all equivalent):
        "stem": [1, 24, 34]
        "stem": {"target": [1, 24, 34], "note": "why"}

    Top-level keys starting with "_" are ignored, so the file can carry its own
    documentation -- which it should. Track ids are not stable across runs (they
    come from the tracker and change with the weights, imgsz or tracker config),
    so a file of bare numbers with no record of what they mean or how they were
    chosen goes stale silently.

    Returns {} when *path* is None or missing, so callers can always call it.
    """
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    overrides = {}
    for stem, value in raw.items():
        if stem.startswith("_"):
            continue                      # documentation, not a video
        if isinstance(value, dict):
            value = value.get("target", [])
        overrides[stem] = [int(track_id) for track_id in value]
    return overrides


def frames_from_track_ids(tracks, track_ids):
    """Union the named tracks into one {frame_idx: (xy, conf)} mapping.

    Earlier ids win on frames claimed by more than one track, so listing the
    subject's own tracklet first makes it authoritative where it overlaps a
    duplicate. Ids that are not present are reported rather than ignored --
    a typo in a hand-written file should be loud, not silently drop frames.
    """
    merged, missing = {}, []
    for track_id in track_ids:
        frames = tracks.get(track_id)
        if frames is None:
            missing.append(track_id)
            continue
        for idx, pose in frames.items():
            merged.setdefault(idx, pose)
    return merged, missing


def summarize_tracks(tracks, conf_threshold=0.2, min_valid_joints=4, limit=6):
    """One-line-per-track summary (longest first) for logging, so a wrong
    target choice is visible in the run output instead of silent."""
    rows = []
    for track_id, frames in tracks.items():
        usable = _usable_frame_count(frames, conf_threshold, min_valid_joints)
        first, last = min(frames), max(frames)
        rows.append((usable, track_id, len(frames), first, last))
    rows.sort(reverse=True)
    return [f"id={track_id} usable={usable} seen={seen} frames[{first}..{last}]"
            for usable, track_id, seen, first, last in rows[:limit]]
