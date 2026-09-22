"""Train the final model on ALL subjects and export it for the live runtime.

The LOSO arms exist to *estimate* performance honestly -- each one holds a subject
out, so none of them has seen the whole corpus. The model that ships should use
every subject, because the measured learning curve is still rising at +0.011
macro-F1 per added participant. This script trains that model once and writes the
artefacts the runtime expects.

Chosen configuration (from the four retrain arms, 15-fold LOSO):
  * 6 classes -- Lift dropped.  macro-F1 0.392 -> 0.476, 15/15 folds, p=0.0001
  * pos_weight capped at 2.0.   lane precision 0.462 -> 0.553, 15/15, p=0.0001
  * NO focal loss.              it lost macro-F1 on 14/15 folds
  * peak targets + mirror augmentation

Exports into --out-dir:
  best_model.pth     weights only (state_dict), the name the runtime looks for
  deploy_model.pt    full bundle: weights + norm stats + config, self-contained
  norm_stats.npz     per-column mean/std from the training data
  config.json        window size, class count, feature layout, trigger settings

The trigger settings matter as much as the weights. The runtime must reproduce
  score = P(Align) x P(Place) x progress, smoothed 3 s, sustained crossing
-- that gate, not the classifier, is what took false alarms from 2.19 to 0.75.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (WindowSet, dedup_augmented, take_key, PANEL_ORDER, task_names)
from models import build
from engine import train_one, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="original")
    ap.add_argument("--aug-dir", default="augmented_mirror")
    ap.add_argument("--out-dir", default="deploy")
    ap.add_argument("--model", default="gru")
    ap.add_argument("--targets", default="peak")
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--hop", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--pos-weight-cap", type=float, default=2.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--keep-lift", action="store_true")
    ap.add_argument("--holdout", default="14,15",
                    help="subjects kept out as a final check; empty uses everything")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    drop_lift = not a.keep_lift
    names = task_names(drop_lift)
    paths = sorted(glob.glob(os.path.join(a.data, "*.pt")))
    aug = dedup_augmented(sorted(glob.glob(os.path.join(a.aug_dir, "*.pt"))))
    hold = {int(x) for x in a.holdout.split(",") if x.strip()}

    tr_p = [p for p in paths if take_key(p)[0] not in hold]
    tr_p += [p for p in aug if take_key(p)[0] not in hold]
    ho_p = [p for p in paths if take_key(p)[0] in hold]
    print("[deploy] %d training files (%d augmented), %d subjects held out %s"
          % (len(tr_p), sum(1 for p in tr_p if "augmented" in p), len(hold),
             sorted(hold) or "(none)"))

    t0 = time.time()
    tr = WindowSet(tr_p, a.win, a.hop, drop_lift=drop_lift)
    Xtr, ytr = tr.tensors()
    # With no holdout there is no clean validation set, so epoch selection falls
    # back to a slice of training data. Stated because it differs from the LOSO
    # arms, which select on a genuinely held-out validation subject.
    if ho_p:
        ho = WindowSet(ho_p, a.win, a.hop, stats=tr.stats, drop_lift=drop_lift)
        Xva, yva = ho.tensors()
    else:
        Xva, yva = Xtr[:2000], {k: v[:2000] for k, v in ytr.items()}
    C = ytr["plateau"].shape[1]
    print("[deploy] %d windows, %d features, %d classes  (%.0fs)"
          % (len(Xtr), Xtr.shape[-1], C, time.time() - t0))

    torch.manual_seed(0)
    model = build(a.model, Xtr.shape[-1], n_tasks=C).to(dev)
    model, best_val, state = train_one(
        model, Xtr, ytr, Xva, yva, dev, a.targets, a.epochs, a.bs,
        pos_weight_cap=(a.pos_weight_cap or None), focal_gamma=a.focal_gamma)

    if ho_p:
        r = evaluate(model, Xva, yva, dev, a.targets)
        print("")
        print("[holdout] macro-F1 %.3f  lane precision %.3f  progress MAE %.1f"
              % (r["macro_f1"], r.get("lane_precision", float("nan")),
                 r["progress_mae"]))
        for k, v in r["per_class_recall"].items():
            print("    %-12s recall %.2f" % (k, v))

    os.makedirs(a.out_dir, exist_ok=True)
    mu, sd = tr.stats
    torch.save(state, os.path.join(a.out_dir, "best_model.pth"))
    np.savez(os.path.join(a.out_dir, "norm_stats.npz"), mean=mu, std=sd)
    torch.save({"state_dict": state, "stats": (mu, sd), "dim": int(Xtr.shape[-1]),
                "n_classes": C, "model": a.model, "targets": a.targets,
                "win": a.win, "hop": a.hop, "class_names": names},
               os.path.join(a.out_dir, "deploy_model.pt"))

    cfg = {
        "architecture": a.model, "hidden_dim": 128, "num_layers": 1,
        "input_dim": int(Xtr.shape[-1]), "window_size": a.win, "hop": a.hop,
        "num_steps": C, "class_names": names,
        "dropped_classes": ([] if a.keep_lift else ["Lift"]),
        "targets": a.targets, "pos_weight_cap": a.pos_weight_cap,
        "focal_gamma": a.focal_gamma,
        "panel_order": PANEL_ORDER,
        "feature_note": ("235 raw columns; the 16 polar_azimuth degree columns are "
                         "replaced by 32 sin/cos columns -> 251. ratios clipped to "
                         "[0,4]. Per-column standardisation with norm_stats.npz, "
                         "computed on the training subjects only."),
        "heads": {"task": "%d sigmoid logits (multi-label, NOT softmax)" % C,
                  "prog": "%d lanes, progress 0-1" % C,
                  "mistake": "1 logit", "bg": "1 logit, no task active"},
        "trigger": {
            "formula": "sigmoid(task)[align] * sigmoid(task)[place] * clip(max(prog),0,1)",
            "align_lane": names.index("Align"),
            "place_lane": names.index("Place"),
            "smooth_windows": 9,
            "smooth_note": "uniform filter over 9 windows = 3 s at hop 10, 30 fps",
            "threshold": 0.06,
            "threshold_note": ("selected on validation subjects; sweep 0.04-0.10 to "
                               "trade detection against false alarms"),
            "sustain_windows": 3,
            "once_per_cycle": True,
            "measured": {"detection": 0.67, "false_alarms_per_event": 0.75,
                         "median_latency_s": -0.3,
                         "baseline_detection": 0.75,
                         "baseline_false_alarms_per_event": 2.19},
        },
        "provenance": {
            "trained_on_subjects": sorted({take_key(p)[0] for p in tr_p}),
            "held_out_subjects": sorted(hold),
            "loso_macro_f1": 0.480, "loso_macro_f1_sd": 0.080,
            "loso_lane_precision": 0.553,
            "note": ("LOSO figures come from bench/loso.py with one subject held out "
                     "per fold. This deployed model trains on more data than any "
                     "single fold, so its true performance should be no worse."),
        },
    }
    json.dump(cfg, open(os.path.join(a.out_dir, "config.json"), "w"), indent=2)

    print("")
    print("[deploy] wrote to %s/" % a.out_dir)
    for f in sorted(os.listdir(a.out_dir)):
        print("    %-20s %8.1f KB"
              % (f, os.path.getsize(os.path.join(a.out_dir, f)) / 1024))
    print("")
    print("To wire into the runtime, copy best_model.pth, norm_stats.npz and")
    print("config.json into hrc_communication/1_recognition/best_model/3d_skeleton/")
    print("")
    print("NOTE: the runtime AssistLSTM has 2 heads and applies softmax over steps.")
    print("This model is a 4-head multi-label GRU, so recognition_manager.py needs")
    print("updating to match -- see the heads and trigger blocks in config.json.")


if __name__ == "__main__":
    main()
