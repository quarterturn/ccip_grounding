#!/usr/bin/env python3
"""
CCIP character grounding with persistent SQLite database tracking.

Usage:
  python ccip_grounder3.py --title "show_name" --reference-dir "reference/char_name" [options]

Features:
  - Maintains a (title).sql database of images and their detected characters.
  - Reference directory name is used as the character tag.
  - Incremental updates: only scans for new files.
  - --clean option to prune missing files from the database.
  - --refresh option to purge existing tags for the character before grounding.
"""
import argparse
import json
import os
import pickle
import sqlite3
import sys
import traceback
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from imgutils.metrics import (
    ccip_extract_feature,
    ccip_batch_extract_features,
    ccip_batch_differences,
    ccip_default_threshold,
)
from common import EXTS, list_images, load_rgb, pad_box, person_crops, head_crop

# ---------------------------------------------------------------- Database
class DBManager:
    """v2 schema: images(path PK, name) + image_tags(path, char) join table.

    v1 stored a comma-separated 'characters' column keyed by basename, which
    caused silent basename collisions and substring (LIKE '%char%') tag bleed.
    Run migrate_v2.py on any v1 database first.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _retry_execute(self, sql, params=None):
        """Execute a SQL statement, retrying on 'database is locked' errors."""
        retries = 0
        while True:
            try:
                return self.conn.execute(sql, params or ())
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and retries < 10:
                    retries += 1
                    time.sleep(0.1)
                else:
                    raise

    def _init_db(self):
        with self.conn:
            self._retry_execute(
                "CREATE TABLE IF NOT EXISTS images ("
                "path TEXT PRIMARY KEY, "
                "name TEXT)"
            )
            self._retry_execute(
                "CREATE TABLE IF NOT EXISTS image_tags ("
                "path TEXT NOT NULL, "
                "char TEXT NOT NULL, "
                "PRIMARY KEY (path, char))"
            )
            self._retry_execute(
                "CREATE INDEX IF NOT EXISTS idx_tags_char ON image_tags(char)")

    def detect_legacy(self):
        """True if the images table still uses the v1 'characters' column."""
        row = self._retry_execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='images'"
        ).fetchone()
        return row is not None and "characters" in (row[0] or "").lower()

    def register_images(self, image_paths):
        """Add new images to the database (keyed by full path)."""
        added = 0
        with self.conn:
            for p in image_paths:
                cur = self._retry_execute(
                    "INSERT OR IGNORE INTO images (path, name) VALUES (?, ?)",
                    (p, os.path.basename(p))
                )
                added += cur.rowcount
        return added

    def add_character_tag(self, path, char_tag):
        """Tag an image with a character (exact match, no duplicates)."""
        with self.conn:
            self._retry_execute(
                "INSERT OR IGNORE INTO image_tags (path, char) VALUES (?, ?)",
                (path, char_tag)
            )

    def remove_character_tag(self, path, char_tag):
        """Remove a character tag from an image."""
        with self.conn:
            self._retry_execute(
                "DELETE FROM image_tags WHERE path = ? AND char = ?",
                (path, char_tag)
            )

    def purge_character(self, char_tag):
        """Remove a specific character tag from all images in the database."""
        with self.conn:
            cur = self._retry_execute(
                "DELETE FROM image_tags WHERE char = ?", (char_tag,))
            return cur.rowcount

    def get_all_images(self):
        cur = self._retry_execute("SELECT path, name FROM images")
        return cur.fetchall()

    def prune_missing(self):
        """Remove entries from DB if the file no longer exists on disk."""
        all_imgs = self.get_all_images()
        removed = 0
        with self.conn:
            for path, _name in all_imgs:
                if not os.path.exists(path):
                    self._retry_execute(
                        "DELETE FROM image_tags WHERE path = ?", (path,))
                    self._retry_execute(
                        "DELETE FROM images WHERE path = ?", (path,))
                    removed += 1
        return removed

    def close(self):
        self.conn.close()

# ---------------------------------------------------------------- utilities
# (list_images, load_rgb, pad_box, person_crops, head_crop live in common.py)

# ---------------------------------------------------------------- references
def build_anchors(ref_dir, cache_path, crop_refs, det_conf, pad, ccip_region):
    paths = list_images(ref_dir, recursive=False)
    if not paths:
        sys.exit(f"CRITICAL: no reference images found in {ref_dir}")
    
    char_name = os.path.basename(os.path.normpath(ref_dir))
    groups = {char_name: paths}

    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    new_cache, anchors, anchor_paths = {}, {}, {}
    for name, p_list in groups.items():
        feats = []
        for p in tqdm(p_list, desc=f"ref [{name}]"):
            key = (os.path.abspath(p), os.path.getmtime(p),
                   os.path.getsize(p), crop_refs, ccip_region)
            if key in cache:
                feat = cache[key]
            else:
                img = load_rgb(p)
                if crop_refs:
                    crops = person_crops(img, det_conf, pad, 0)
                    img = max(crops, key=lambda c: c[2].size[0] * c[2].size[1])[2]
                if ccip_region == "head":
                    img = head_crop(img, 0.25)
                feat = np.asarray(ccip_extract_feature(img))
            new_cache[key] = feat
            feats.append(feat)
        anchors[name] = feats
        anchor_paths[name] = p_list
        print(f"  {name}: {len(feats)} reference features")

    with open(cache_path, "wb") as f:
        pickle.dump(new_cache, f)
    return anchors, anchor_paths

# ---------------------------------------------------------------- CLIP
class ClipChecker:
    def __init__(self, model_name, pairs, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.pairs = pairs
        self.texts = [t for pair in pairs for t in pair]
        self.score(Image.new("RGB", (224, 224), (255, 255, 255)))

    def score(self, crop):
        torch = self.torch
        with torch.no_grad():
            inputs = self.proc(text=self.texts, images=crop,
                               return_tensors="pt", padding=True).to(self.device)
            out = self.model(**inputs)
            logits = out.logits_per_image[0]
        return [float(logits[2 * i:2 * i + 2].softmax(-1)[0])
                for i in range(len(self.pairs))]

def parse_pairs(raw):
    pairs = []
    for s in raw or []:
        if "|" not in s:
            sys.exit(f'--clip-pair must look like "yes prompt|no prompt", got: {s}')
        yes, no = (x.strip() for x in s.split("|", 1))
        pairs.append((yes, no))
    return pairs

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="CCIP character grounder + SQLite identity index")
    base = "/home/alex/Documents/ccip_grounding"
    
    ap.add_argument("--title", required=True, help="Title of the show (used for DB name)")
    ap.add_argument("--reference-dir", required=True, help="Path to character reference folder")
    ap.add_argument("--input", default=f"{base}/input")
    ap.add_argument("--output", default=f"{base}/output")
    ap.add_argument("--recursive", action="store_true", help="search INPUT subfolders")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--crop-refs", action="store_true")
    ap.add_argument("--ccip-region", choices=["person", "head"], default="person")
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--pad", type=float, default=0.1)
    ap.add_argument("--min-crop", type=int, default=64)
    ap.add_argument("--clip-pair", action="append")
    ap.add_argument("--clip-min", type=float, default=0.6)
    ap.add_argument("--clip-region", choices=["head", "person"], default="head")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--clean", action="store_true", help="Prune missing images from DB")
    ap.add_argument("--yes", action="store_true", help="Skip --clean confirmation (for scripts)")
    ap.add_argument("--refresh", action="store_true", help="Purge existing tags for the character before grounding")
    ap.add_argument("--gpu", type=int, default=None,
                    help="Pin all GPU work (ONNX + CLIP) to this GPU ID "
                         "(sets CUDA_VISIBLE_DEVICES before any model loads)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Images per batched CCIP extraction/difference call (default: 32)")

    args = ap.parse_args()

    # Must happen before onnxruntime/torch create a CUDA context.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Pinning GPU work to physical GPU {args.gpu}")

    ref_dir = os.path.abspath(args.reference_dir)
    # removed strict reference-dir path check to allow external reference directories

    char_tag = os.path.basename(os.path.normpath(ref_dir))
    db_path = os.path.join(base, f"{args.title}.sql")
    db = DBManager(db_path)
    if db.detect_legacy():
        sys.exit(f"ERROR: {db_path} uses the legacy v1 schema (comma-separated "
                 "'characters' column keyed by basename).\n"
                 f"Run: python3 migrate_v2.py --db {db_path} --dry-run  (then without --dry-run)")

    if args.clean:
        print("Cleaning database...")
        if sys.stdin.isatty() and not args.yes:
            confirm = input("WARNING: This will delete entries for images no longer on disk. Proceed? (y/N): ")
            ok = confirm.lower() == 'y'
        else:
            ok = bool(args.yes)
            if not ok:
                print("Non-interactive mode without --yes; skipping cleanup.")
        if ok:
            removed = db.prune_missing()
            print(f"Removed {removed} stale entries.")
        elif sys.stdin.isatty():
            print("Cleanup cancelled.")

    os.makedirs(args.output, exist_ok=True)
    threshold = args.threshold if args.threshold is not None else float(ccip_default_threshold())
    print(f"CCIP difference threshold: {threshold:.4f}")
    print(f"CCIP region: {args.ccip_region}")

    anchors, anchor_paths = build_anchors(
        ref_dir, os.path.join(args.output, ".ref_cache.pkl"),
        args.crop_refs, args.det_conf, args.pad, args.ccip_region)
    
    char_names = list(anchors.keys()) 
    ref_feats, ref_owner, ref_paths = [], [], []
    for ci, name in enumerate(char_names):
        ref_feats += anchors[name]
        ref_owner += [ci] * len(anchors[name])
        ref_paths += anchor_paths[name]
    ref_owner = np.array(ref_owner)
    n_ref = len(ref_feats)

    if args.refresh:
        print(f"Refreshing identity index for {char_tag}...")
        removed = db.purge_character(char_tag)
        print(f"Removed {removed} existing tags for {char_tag}.")

    print("Updating identity index...")
    input_imgs = list_images(args.input, args.recursive)

    added_input = db.register_images(input_imgs)
    # Reference images are used for anchors but should not be processed as input images
    print(f"Registered {added_input} input images. (Reference images are used as anchors only)")

    pairs = parse_pairs(args.clip_pair)
    clip = None
    if pairs:
        print(f"Loading CLIP: {args.clip_model}")
        clip = ClipChecker(args.clip_model, pairs)
        for yes, no in pairs:
            print(f"  require P('{yes}') >= {args.clip_min}  vs  '{no}'")

    all_db_images = db.get_all_images()
    if not all_db_images:
        sys.exit("No images found in database to process.")
    
    print(f"Processing {len(all_db_images)} images from database")
    results_path = os.path.join(args.output, f"results_{char_tag}.jsonl")
    counts = {"MATCH": 0, "DIST_FAIL": 0, "CLIP_FAIL": 0, "CLIP_ERROR": 0, "LOAD_ERROR": 0}
    clip_error_shown = False

    def write_rec(fh, rec):
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def score_and_log(log, path, crops, diffs):
        """Threshold/CLIP-gate one image's crops against the ref matrix."""
        nonlocal clip_error_shown
        best_per_char = {}
        rec = {"image": path, "matches": [], "crops": []}
        for k, (box, det_score, _crop) in enumerate(crops):
            char_scores = []
            for ci in range(len(char_names)):
                d = np.sort(diffs[k, ref_owner == ci])
                char_scores.append(float(d[:max(1, min(args.topk, len(d)))].mean()))

            ci_best = int(np.argmin(char_scores))
            s_best = char_scores[ci_best]
            nearest = int(np.argmin(diffs[k]))

            rec["crops"].append({"box": list(box), "det_score": det_score,
                                 "best_char": char_names[ci_best], "ccip_diff": s_best,
                                 "nearest_ref": os.path.basename(ref_paths[nearest])})
            if s_best <= threshold:
                if ci_best not in best_per_char or s_best < best_per_char[ci_best][0]:
                    best_per_char[ci_best] = (s_best, k)

        if not best_per_char:
            rec["status"] = "DIST_FAIL"
            counts["DIST_FAIL"] += 1
            write_rec(log, rec)
            return

        any_match = False
        statuses = []
        for ci, (score, k) in best_per_char.items():
            name = char_names[ci]
            box, _ds, crop = crops[k]
            m = {"char": name, "box": list(box), "ccip_diff": score}

            if clip:
                region = head_crop(crop, 0.25) if args.clip_region == "head" else crop
                try:
                    probs = clip.score(region)
                except Exception as e:
                    if not clip_error_shown:
                        traceback.print_exc()
                        clip_error_shown = True
                    m["status"] = "CLIP_ERROR"
                    m["error"] = repr(e)
                    statuses.append("CLIP_ERROR")
                    rec["matches"].append(m)
                    continue
                m["clip"] = {yes: p for (yes, _no), p in zip(pairs, probs)}
                if min(probs) < args.clip_min:
                    m["status"] = "CLIP_FAIL"
                    statuses.append("CLIP_FAIL")
                    rec["matches"].append(m)
                    continue

            m["status"] = "MATCH"
            statuses.append("MATCH")
            any_match = True
            db.add_character_tag(path, name)

            rec["matches"].append(m)

        if any_match:
            rec["status"] = "MATCH"
        elif "CLIP_ERROR" in statuses:
            rec["status"] = "CLIP_ERROR"
        else:
            rec["status"] = "CLIP_FAIL"
        counts[rec["status"]] += 1
        write_rec(log, rec)

    with open(results_path, "w") as log:
        for bstart in tqdm(range(0, len(all_db_images), args.batch_size),
                           desc="batches"):
            batch = all_db_images[bstart:bstart + args.batch_size]

            # Stage 1: load + detect per image (failures isolated per image)
            batch_items = []  # (path, crops)
            for path, img_name in batch:
                try:
                    img = load_rgb(path)
                    crops = person_crops(img, args.det_conf, args.pad, args.min_crop)
                    batch_items.append((path, crops))
                except Exception as e:
                    rec = {"image": path, "matches": [], "crops": [],
                           "status": "LOAD_ERROR", "error": repr(e)}
                    counts["LOAD_ERROR"] += 1
                    write_rec(log, rec)

            if not batch_items:
                continue

            # Stage 2: ONE batched CCIP extraction for every crop in the batch
            all_crops = [c for _p, cs in batch_items for c in cs]
            crop_imgs = [head_crop(c[2], 0.25) if args.ccip_region == "head" else c[2]
                         for c in all_crops]
            try:
                crop_feats = [np.asarray(f) for f in
                              ccip_batch_extract_features(crop_imgs)]
            except Exception as e:
                # batched extraction failed (e.g. one bad crop): fall back to
                # per-image processing so only the truly broken image errors
                crop_feats = None
            if crop_feats is None:
                for path, crops in batch_items:
                    try:
                        crop_feats_i = [np.asarray(ccip_extract_feature(
                            head_crop(c[2], 0.25) if args.ccip_region == "head" else c[2]))
                            for c in crops]
                        full = np.asarray(ccip_batch_differences(ref_feats + crop_feats_i))
                        diffs = full[n_ref:, :n_ref]
                    except Exception as e2:
                        rec = {"image": path, "matches": [], "crops": [],
                               "status": "LOAD_ERROR", "error": repr(e2)}
                        counts["LOAD_ERROR"] += 1
                        write_rec(log, rec)
                        continue
                    score_and_log(log, path, crops, diffs)
                continue

            # Stage 3: ONE difference matrix for the whole batch
            try:
                full = np.asarray(ccip_batch_differences(ref_feats + crop_feats))
                diffs = full[n_ref:, :n_ref]
            except Exception as e:
                # whole-batch diff failed: log all batch images as errors
                for path, _cs in batch_items:
                    rec = {"image": path, "matches": [], "crops": [],
                           "status": "LOAD_ERROR", "error": repr(e)}
                    counts["LOAD_ERROR"] += 1
                    write_rec(log, rec)
                continue

            # Stage 4: split the matrix back per image
            off = 0
            for path, crops in batch_items:
                n = len(crops)
                score_and_log(log, path, crops, diffs[off:off + n])
                off += n

    print("\nDone.")
    for k, v in counts.items():
        print(f"  {k:<11} {v}")
    print(f"Log: {results_path}")
    db.close()

if __name__ == "__main__":
    main()
