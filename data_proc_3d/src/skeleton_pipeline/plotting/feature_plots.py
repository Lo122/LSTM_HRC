"""Per-panel feature plots -- one PNG per panel group (see
skeleton_pipeline/features/h36m_features.py's compute_all_features), each
showing every column in that panel as a labeled time-series line. Mirrors
data_proc_2d's panel_titles/feature_dataframes grouping (one plot per
feature GROUP, e.g. all 16 joints' speed together on one "Joint Speed"
plot), not one plot per individual scalar column -- 181 tiny individual
plots would be far less useful than ~16 grouped ones for actually reading
these features. plot_panels also writes one extra "_overview.png" with every
panel group laid out as a subplot on a single image, for a quick-look
dashboard of the whole feature set (see plot_panels' include_overview).

plot_overview_with_labels() is that same overview with the annotated step
ribbon and task_progress curve underneath, on a shared time axis -- for
checking the labels against the motion they describe. It is called from
app/build_training_pairs.py, the one stage that holds features and labels at
once; the plain overview is what the generation stage writes, before any
annotation is loaded.

Deliberately matplotlib (not the fast plain-OpenCV renderer in
skeleton_pipeline/render/) -- these are a handful of one-shot static plots
per video, not a per-frame video overlay, so matplotlib's ~100ms/plot cost
(see skeleton_pipeline/render/skeleton_video.py's docstring) is a non-issue
here.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- these are saved-to-disk plots, no display
import matplotlib.pyplot as plt


def _slug(title):
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def plot_panels(feature_dict, panel_groups, timestamps, output_dir, prefix,
                 max_lines_per_plot=20, include_overview=True):
    """feature_dict/panel_groups: see compute_all_features()'s return value.
    timestamps: (T,) seconds. Writes one PNG per panel to
    output_dir/<prefix>_<panel_slug>.png. Returns the list of written paths.

    max_lines_per_plot: panels with more columns than this are split across
    multiple figures (same panel title, "(1/2)" etc. suffix) so the legend
    stays readable -- none of this project's current panels need it (at
    most 16 joints/panel), but keeps this robust if that ever changes.

    include_overview: also write one extra output_dir/<prefix>_overview.png
    with every panel group as its own subplot on a single image -- a
    quick-look dashboard of the whole feature set for one video. Complements
    (doesn't replace) the per-panel PNGs above, which are what you want for
    actually reading line-by-line values/labels.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for panel_title, columns in panel_groups.items():
        n_chunks = max(1, (len(columns) + max_lines_per_plot - 1) // max_lines_per_plot)
        for chunk_idx in range(n_chunks):
            chunk_columns = columns[chunk_idx * max_lines_per_plot:(chunk_idx + 1) * max_lines_per_plot]
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in chunk_columns:
                ax.plot(timestamps, feature_dict[col], label=col, linewidth=1.0)
            title = panel_title if n_chunks == 1 else f"{panel_title} ({chunk_idx + 1}/{n_chunks})"
            ax.set_title(title)
            ax.set_xlabel("time (s)")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            suffix = _slug(panel_title) if n_chunks == 1 else f"{_slug(panel_title)}_{chunk_idx + 1}"
            out_path = output_dir / f"{prefix}_{suffix}.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            written.append(out_path)

    if include_overview and panel_groups:
        written.append(_plot_overview(feature_dict, panel_groups, timestamps, output_dir, prefix))
    return written


def _plot_overview(feature_dict, panel_groups, timestamps, output_dir, prefix):
    """One combined PNG with every panel group as its own subplot -- lines
    are unlabeled here (no per-column legend, to keep a dozen-plus subplots
    readable); see the per-panel PNGs from plot_panels for labeled detail."""
    panel_titles = list(panel_groups.keys())
    n_panels = len(panel_titles)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)
    _draw_panel_grid(axes, feature_dict, panel_groups, timestamps, n_cols, n_rows)

    fig.suptitle(f"{prefix} -- feature overview", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = output_dir / f"{prefix}_overview.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _draw_panel_grid(axes, feature_dict, panel_groups, timestamps, n_cols, n_rows):
    """Fills a (n_rows, n_cols) axes array with one panel group each, blanking
    any leftover cells. Shared by the plain overview and the labelled one."""
    panel_titles = list(panel_groups.keys())
    for idx, panel_title in enumerate(panel_titles):
        ax = axes[idx // n_cols][idx % n_cols]
        for col in panel_groups[panel_title]:
            ax.plot(timestamps, feature_dict[col], linewidth=0.8)
        ax.set_title(panel_title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("time (s)")

    for idx in range(len(panel_titles), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")


def plot_overview_with_labels(feature_dict, panel_groups, timestamps, output_dir, prefix,
                               label_vector, label_names, task_progress, fps,
                               output_name=None, mistake=None):
    """The feature overview with the label track drawn underneath it, so a
    feature's behaviour can be read against the label it was annotated as.

    Laid out as one full-width row per panel, stacked, every row sharing one
    time axis (seconds) with the label rows at the bottom:

      - one LANE PER LABEL, each showing only that label's own spans
      - the mistake flag, when *mistake* (a (T,) 0/1 array) is given
      - task_progress, the 0-100% curve for the frame's task

    A single column rather than _plot_overview's 3-wide grid: the point here is
    reading a feature against the label it happened in, which needs every row to
    line up vertically at the same instant. _plot_overview keeps the grid, which
    fits more panels on screen when there are no labels to compare against.

    label_vector: (T, n_labels) per-label scores over FRAMES -- the *_vector
    output of skeleton_pipeline.dataset.labels.extract_labels (use a _plateau
    one, which is flat 1.0 across a span rather than peaked at its midpoint).
    A lane rather than a single argmax ribbon because labels OVERLAP: ~21% of
    annotated frames carry two at once (Lift + Place, Lift + Align), and an
    argmax strip can only ever show the winner, hiding the concurrency that
    makes those frames interesting.

    label_names: one name per column of label_vector, in column order. Passed in
    rather than imported so this module stays independent of the label taxonomy.

    task_progress/mistake: (T,) arrays over frames. Converted to seconds with
    *fps* here; features and labels are otherwise indexed the same way.
    """
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_titles = list(panel_groups.keys())
    n_panels = len(panel_titles)
    n_labels = np.asarray(label_vector).shape[1]

    # Explicit margins instead of tight_layout: the label lanes' spans are not
    # something tight_layout measures correctly, and it responds by leaving most
    # of the canvas blank. Margins in inches, converted to the figure fractions
    # gridspec wants, so they stay constant however many panels there are.
    lane_height = max(0.3 * n_labels + 0.5, 1.3)
    mistake_rows = [0.55] if mistake is not None else []
    fig_height = 1.7 * n_panels + 3.0 + lane_height + sum(mistake_rows)
    fig = plt.figure(figsize=(16, fig_height))
    grid = fig.add_gridspec(n_panels + 2 + len(mistake_rows), 1,
                            height_ratios=[1.7] * n_panels + [lane_height] + mistake_rows + [1.6],
                            hspace=0.45,
                            left=0.115, right=0.99,
                            top=1 - 0.75 / fig_height, bottom=0.85 / fig_height)

    first_ax = None
    for idx, panel_title in enumerate(panel_titles):
        ax = fig.add_subplot(grid[idx, 0], sharex=first_ax)
        first_ax = first_ax or ax
        for col in panel_groups[panel_title]:
            ax.plot(timestamps, feature_dict[col], linewidth=0.8)
        # Title on the left rather than centred: with this many stacked rows a
        # centred title reads as belonging to no row in particular.
        ax.set_title(panel_title, fontsize=9, loc="left", pad=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False, labelsize=8)

    lanes_ax = fig.add_subplot(grid[n_panels, 0], sharex=first_ax)
    _draw_label_lanes(lanes_ax, label_vector, label_names, fps)

    if mistake is not None:
        mistake_ax = fig.add_subplot(grid[n_panels + 1, 0], sharex=first_ax)
        _draw_mistake(mistake_ax, mistake, fps)

    progress_ax = fig.add_subplot(grid[n_panels + 1 + len(mistake_rows), 0], sharex=first_ax)
    _draw_progress(progress_ax, task_progress, fps)

    # One x range for every row, so a time read off any feature lines up with
    # the lanes -- including where the labels stop short of the video's end.
    if len(timestamps):
        (first_ax or progress_ax).set_xlim(float(timestamps[0]), float(timestamps[-1]))

    fig.suptitle(f"{prefix} -- feature overview with labels", fontsize=12,
                 y=1 - 0.28 / fig_height)

    out_path = output_dir / (output_name or f"{prefix}_overview_labeled.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _true_runs(flags):
    """[(start, end), ...] index ranges of the True runs in a 1-D bool array."""
    import numpy as np

    flags = np.asarray(flags, dtype=bool)
    if not flags.size:
        return []
    edges = np.flatnonzero(np.diff(flags.astype(np.int8)))
    starts = np.concatenate(([0], edges + 1))
    ends = np.concatenate((edges + 1, [flags.size]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if flags[s]]


def _draw_label_lanes(ax, label_vector, label_names, fps, active_threshold=0.5):
    """One horizontal lane per label, each carrying only that label's spans.

    Every lane is drawn independently, so frames where two labels are annotated
    at once show a band in both lanes -- which a single argmax strip cannot do,
    since it has one value per frame and must drop the loser. The lane names sit
    on the y axis, so no legend row is needed.
    """
    import numpy as np

    scores = np.asarray(label_vector, dtype=float)
    colormap = plt.get_cmap("tab20")

    for column, name in enumerate(label_names):
        colour = colormap(column % 20)
        y = len(label_names) - 1 - column      # first label on top, reading order
        for start, end in _true_runs(scores[:, column] >= active_threshold):
            ax.broken_barh([(start / fps, (end - start) / fps)], (y - 0.4, 0.8),
                           facecolors=colour)

    ax.set_ylim(-0.6, len(label_names) - 0.4)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(list(label_names)[::-1], fontsize=8)
    ax.set_title("annotated labels (one lane each; overlaps visible)",
                 fontsize=9, loc="left", pad=3)
    ax.grid(True, axis="x", alpha=0.3)
    # The progress row shares this x axis and carries the tick labels.
    ax.tick_params(labelbottom=False)


def _draw_mistake(ax, mistake, fps):
    """The binary mistake flag as red bands, on its own thin row: it coincides
    with whichever task lane is active (a mistake IS one of the tasks, done
    wrong), so it is a separate row rather than a colour inside the lanes."""
    import numpy as np

    flag = np.asarray(mistake).astype(bool)
    for start, end in _true_runs(flag):
        # Hatched, not just red: one of the task lanes above is also drawn red
        # by the colormap, and this row must not read as another task.
        ax.axvspan(start / fps, end / fps, facecolor="tab:red", alpha=0.75,
                   hatch="///", edgecolor="black", linewidth=0.0)

    ax.set_yticks([])
    ax.set_ylabel("mistake", fontsize=9)
    ax.set_title(f"mistake  ({100.0 * flag.mean():.1f}% of frames)" if flag.size else "mistake",
                 fontsize=9, loc="left", pad=3)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelbottom=False)


def _draw_progress(ax, task_progress, fps):
    import numpy as np

    task_progress = np.asarray(task_progress, dtype=float)
    seconds = np.arange(task_progress.size) / fps
    ax.plot(seconds, task_progress, linewidth=0.9, color="tab:blue")
    ax.set_ylim(-5, 105)
    ax.set_ylabel("%", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_title("task_progress", fontsize=9, loc="left", pad=3)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
