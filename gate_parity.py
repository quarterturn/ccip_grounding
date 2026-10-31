#!/usr/bin/env python3
"""A/B gate parity: facesaver's AGPL YOLO gate vs MIT DeepGHS face detector.

Purpose: the product build cannot ship Ultralytics/yolov8x6_animeface
(AGPL-3.0).  Measure whether the MIT-licensed deepghs/anime_face_detection
model (face_detect_v1.4_s via imgutils ONNX, already the license family the
grounding crop stage uses) makes the same keep/drop decisions as the
facesaver gate.  Agreement is measured on gate *decisions*, not raw boxes:
both detectors' boxes pass through facesaver's exact geometry gate
(face_qualifies() reproduced verbatim from anime-facesaver v-main).

Two corpora answer two different questions:
  --db grounding.db   the stills that passed the historical YOLO gate.
                      Measures the keep side: what fraction of YOLO's
                      verdicts DeepGHS reproduces at each threshold.
  --images raw_frames/ uniformly sampled RAW frames that never met any
                      gate.  Measures both sides independently: yolo_only
                      frames are DeepGHS misses, dg_only frames are new
                      still candidates YOLO never kept.

DeepGHS runs once at the lowest sweep threshold; higher thresholds are
re-gated in-process from the recorded scores.  This is exact: NMS never
lets a low-score box suppress a higher-score one, so kept@t equals
[kept@min_t with score >= t].  The same argument lets YOLO run once at
its canonical conf and stay the fixed baseline.

CUDA note: set CUDA_VISIBLE_DEVICES (--gpu) BEFORE any CUDA-using import
and warm the context before importing ultralytics -- ultralytics rewrites
CVD on first predict() when no context exists, silently un-pinning GPU1
work onto GPU0.

Outputs in --out: decisions.csv, summary.txt, evidence/<bucket>/ frames,
<bucket>_all.txt full disagreement lists.
"""

import argparse
import csv
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", help="grounding DB (v2) whose images.path list "
                                  "is the corpus (historical-gate side)")
    src.add_argument("--images", help="directory of images (raw sampled frames)")
    p.add_argument("--out", help="output dir (default: <input>_gate_parity)")
    p.add_argument("--yolo-model", default=os.path.expanduser(
        "~/Documents/anime-facesaver/models/yolov8x6_animeface.pt"))
    p.add_argument("--yolo-conf", type=float, default=0.3,
                   help="production facesaver gate conf")
    p.add_argument("--yolo-iou", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=1280,
                   help="production inference size for yolov8x6_animeface")
    p.add_argument("--dg-level", default="s", choices=("n", "s"),
                   help="deepghs face_detect size (v1.4)")
    p.add_argument("--dg-version", default="v1.4")
    p.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,"
                                           "0.40,0.50,0.60,0.70")
    p.add_argument("--headline", type=float, default=0.30,
                   help="threshold whose disagreements get evidence copies")
    # geometry gate: facesaver defaults
    p.add_argument("--min-width", type=int, default=100)
    p.add_argument("--min-height", type=int, default=100)
    p.add_argument("--min-face-fraction", type=float, default=0.005)
    p.add_argument("--max-face-fraction", type=float, default=0.30)
    p.add_argument("--min-aspect", type=float, default=0.45)
    p.add_argument("--max-aspect", type=float, default=1.6)
    p.add_argument("--edge-margin", type=int, default=0)
    p.add_argument("--path-map", action="append", default=[],
                   metavar="OLD=NEW", help="rewrite corpus paths (repeatable); "
                   "e.g. /mnt/md0/image-training-datasets="
                   "/home/alex/Documents/image-training-datasets on .80")
    p.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES pin (e.g. 1)")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--evidence", type=int, default=40,
                   help="max frames copied per disagreement bucket (0=off)")
    return p.parse_args()


# --- facesaver geometry gate (verbatim from anime_facesaver.py, v-main;
#     credit: makoto/anime-facesaver) ----------------------------------------
def face_qualifies(box, frame_shape, min_width, min_height,
                   min_face_fraction, max_face_fraction,
                   min_aspect, max_aspect, edge_margin):
    x1, y1, x2, y2 = box
    frame_h, frame_w = frame_shape[:2]

    w, h = x2 - x1, y2 - y1
    if w < min_width or h < min_height:
        return False

    area_frac = (w * h) / float(frame_w * frame_h)
    if min_face_fraction > 0 and area_frac < min_face_fraction:
        return False
    if max_face_fraction > 0 and area_frac > max_face_fraction:
        return False

    if h > 0:
        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            return False

    if edge_margin > 0:
        if (x1 < edge_margin or y1 < edge_margin
                or x2 > frame_w - edge_margin
                or y2 > frame_h - edge_margin):
            return False

    return True
# -----------------------------------------------------------------------------


def gate_kwargs(args):
    return dict(min_width=args.min_width, min_height=args.min_height,
                min_face_fraction=args.min_face_fraction,
                max_face_fraction=args.max_face_fraction,
                min_aspect=args.min_aspect, max_aspect=args.max_aspect,
                edge_margin=args.edge_margin)


def db_paths(db_path):
    """images.path from a v2 DB.  Copied aside first: the dataset dirs keep
    stale -wal/-shm siblings and we must never touch them read-write."""
    import shutil
    import sqlite3
    import tempfile
    tmp = tempfile.mkdtemp(prefix="gate_parity_db_")
    for suffix in ("", "-wal", "-shm"):
        src = db_path + suffix
        if os.path.exists(src):
            shutil.copy(src, tmp)
    local = os.path.join(tmp, os.path.basename(db_path))
    con = sqlite3.connect(local)
    paths = [r[0] for r in con.execute(
        "select path from images order by path")]
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return paths


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.db:
        paths = db_paths(args.db)
        default_out = args.db + "_gate_parity"
    else:
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        default_out = os.path.join(
            os.path.dirname(os.path.abspath(args.images)),
            os.path.basename(os.path.abspath(args.images)) + "_gate_parity")
        paths = sorted(
            os.path.join(args.images, f) for f in os.listdir(args.images)
            if os.path.splitext(f)[1].lower() in exts)
    if args.limit:
        paths = paths[: args.limit]
    for pm in args.path_map:
        old, new = pm.split("=", 1)
        paths = [new + p[len(old):] if p.startswith(old) else p
                 for p in paths]
    n_remap_missing = sum(1 for p in paths if not os.path.exists(p))
    if n_remap_missing == len(paths):
        sys.exit(f"All {len(paths)} corpus paths unreadable from this host "
                 f"(first: {paths[0]}). Use --path-map for this machine.")
    out = args.out or default_out.rstrip("/") + ""
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "evidence", "yolo_only"), exist_ok=True)
    os.makedirs(os.path.join(out, "evidence", "dg_only"), exist_ok=True)

    thresholds = sorted(float(t) for t in args.thresholds.split(","))
    min_t = min(thresholds)
    gk = gate_kwargs(args)

    # CUDA warm BEFORE ultralytics import (see module docstring).
    dev = args.device
    if dev == "cuda":
        import torch
        torch.zeros(1, device="cuda")
    from ultralytics import YOLO
    from imgutils.detect import detect_faces
    import cv2
    from PIL import Image

    yolo = YOLO(args.yolo_model)

    rows = []
    n_bad = 0
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        img = cv2.imread(path)
        if img is None:
            n_bad += 1
            continue

        res = yolo.predict(img, conf=args.yolo_conf, iou=args.yolo_iou,
                           imgsz=args.imgsz, device=dev, verbose=False)
        yboxes = []
        for r in res:
            if r.boxes is None:
                continue
            for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().numpy()):
                yboxes.append((xyxy, float(c)))
        y_ok = [s for b, s in yboxes if face_qualifies(b, img.shape, **gk)]

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dets = detect_faces(pil, level=args.dg_level,
                            version=args.dg_version,
                            conf_threshold=min_t)
        d_scores = [(b, s) for b, _lbl, s in dets
                    if face_qualifies(b, img.shape, **gk)]

        row = {"path": path,
               "yolo_score": round(max(y_ok), 4) if y_ok else "",
               "yolo_keep": int(bool(y_ok))}
        for t in thresholds:
            kept = [s for b, s in d_scores if s >= t]
            row[f"dg_keep@{t:.2f}"] = int(bool(kept))
            row[f"dg_score@{t:.2f}"] = (round(max(kept), 4)
                                        if kept else "")
        rows.append(row)

        if i % 100 == 0 or i == len(paths):
            el = time.time() - t0
            print(f"[{i}/{len(paths)}] {el / 60:.1f} min "
                  f"({el / i:.2f} s/frame)", flush=True)

    # --- outputs -------------------------------------------------------------
    import json
    with open(os.path.join(out, "decisions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [f"frames={len(rows)} unreadable={n_bad}",
             f"yolo baseline: {args.yolo_model} conf={args.yolo_conf} "
             f"imgsz={args.imgsz} (production facesaver gate params)",
             f"deepghs: face_detect_{args.dg_version}_{args.dg_level} "
             f"(MIT, deepghs/anime_face_detection)  "
             f"card-recommended threshold: 0.307", ""]
    ykeep = [r["yolo_keep"] for r in rows]
    lines.append(f"YOLO keep rate: {sum(ykeep)}/{len(rows)}")
    lines.append("")
    lines.append(f"{'t':>5} {'both':>6} {'yolo_only':>9} {'dg_only':>8} "
                 f"{'neither':>7} {'agree%':>7} {'dg_keep%':>8} "
                 f"{'recall_vs_yolo%':>15}")
    stats = {}
    for t in thresholds:
        k = f"dg_keep@{t:.2f}"
        both = sum(1 for r in rows if r["yolo_keep"] and r[k])
        yonly = sum(1 for r in rows if r["yolo_keep"] and not r[k])
        donly = sum(1 for r in rows if not r["yolo_keep"] and r[k])
        neither = sum(1 for r in rows if not r["yolo_keep"] and not r[k])
        agree = 100.0 * (both + neither) / len(rows)
        recall = 100.0 * both / (both + yonly) if both + yonly else 0.0
        dgk = 100.0 * (both + donly) / len(rows)
        stats[f"{t:.2f}"] = dict(both=both, yolo_only=yonly, dg_only=donly,
                                 neither=neither)
        lines.append(f"{t:>5.2f} {both:>6} {yonly:>9} {donly:>8} "
                     f"{neither:>7} {agree:>7.1f} {dgk:>8.1f} {recall:>15.1f}")

    # headline disagreement lists + evidence copies
    hk = f"dg_keep@{args.headline:.2f}"
    y_only = [r for r in rows if r["yolo_keep"] and not r[hk]]
    d_only = [r for r in rows if not r["yolo_keep"] and r[hk]]
    for name, bucket, sortkey in (("yolo_only", y_only,
                                   lambda r: r["yolo_score"] or 0),
                                  ("dg_only", d_only,
                                   lambda r: r[hk.replace('keep', 'score')]
                                   or 0)):
        with open(os.path.join(out, f"{name}_all.txt"), "w") as f:
            f.write("\n".join(r["path"] for r in
                              sorted(bucket, key=sortkey, reverse=True)))
        if args.evidence:
            for r in sorted(bucket, key=sortkey, reverse=True)[: args.evidence]:
                base = os.path.basename(r["path"])
                sc = r["yolo_score"] if name == "yolo_only" else \
                    r[hk.replace("keep", "score")]
                dst = os.path.join(out, "evidence", name,
                                   f"{sc or 0:.3f}__{base}")
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(r["path"], dst)
    lines.append("")
    lines.append(f"headline t={args.headline:.2f}: yolo_only "
                 f"{len(y_only)} | dg_only {len(d_only)} "
                 f"(lists in *_all.txt, copies in evidence/)")
    summary = "\n".join(lines) + "\n"
    print(summary)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(summary)
    with open(os.path.join(out, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print("GATE PARITY DONE")


if __name__ == "__main__":
    main()
