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
from PIL import Image
from tqdm import tqdm

from imgutils.detect import detect_person, detect_heads
from imgutils.metrics import (
    ccip_extract_feature,
    ccip_batch_differences,
    ccip_default_threshold,
)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ---------------------------------------------------------------- Database
class DBManager:
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
                "name TEXT PRIMARY KEY, "
                "path TEXT, "
                "characters TEXT)" # Comma-separated list
            )

    def register_images(self, image_paths):
        """Add new images to the database."""
        added = 0
        with self.conn:
            for p in image_paths:
                name = os.path.basename(p)
                try:
                    self._retry_execute(
                        "INSERT INTO images (name, path, characters) VALUES (?, ?, ?)",
                        (name, p, "")
                    )
                    added += 1
                except sqlite3.IntegrityError:
                    pass
        return added

    def add_character_tag(self, image_name, char_tag):
        """Add a character tag to an image, avoiding duplicates."""
        with self.conn:
            row = self._retry_execute(
                "SELECT characters FROM images WHERE name = ?", (image_name,)
            ).fetchone()
            if row:
                chars = row[0].split(",") if row[0] else []
                if char_tag not in chars:
                    chars.append(char_tag)
                    self._retry_execute(
                        "UPDATE images SET characters = ? WHERE name = ?",
                        (",".join(chars), image_name)
                    )

    def remove_character_tag(self, image_name, char_tag):
        """Remove a character tag from an image."""
        with self.conn:
            row = self._retry_execute(
                "SELECT characters FROM images WHERE name = ?", (image_name,)
            ).fetchone()
            if row:
                chars = row[0].split(",") if row[0] else []
                if char_tag in chars:
                    chars.remove(char_tag)
                    self._retry_execute(
                        "UPDATE images SET characters = ? WHERE name = ?",
                        (",".join(chars), image_name)
                    )

    def purge_character(self, char_tag):
        """Remove a specific character tag from all images in the database."""
        with self.conn:
            cursor = self._retry_execute("SELECT name, characters FROM images")
            all_rows = cursor.fetchall()
            updated = 0
            for name, characters in all_rows:
                chars = characters.split(",") if characters else []
                if char_tag in chars:
                    chars.remove(char_tag)
                    self._retry_execute(
                        "UPDATE images SET characters = ? WHERE name = ?",
                        (",".join(chars), name)
                    )
                    updated += 1
        return updated

    def get_all_images(self):
        cursor = self._retry_execute("SELECT name, path FROM images")
        return cursor.fetchall()

    def prune_missing(self):
        """Remove entries from DB if the file no longer exists on disk."""
        all_imgs = self.get_all_images()
        removed = 0
        with self.conn:
            for name, path in all_imgs:
                if not os.path.exists(path):
                    self._retry_execute("DELETE FROM images WHERE name = ?", (name,))
                    removed += 1
        return removed

    def close(self):
        self.conn.close()

# ---------------------------------------------------------------- utilities
def list_images(directory, recursive):
    if not os.path.isdir(directory):
        return []
    out = []
    if recursive:
        for root, _, files in os.walk(directory):
            out += [os.path.join(root, f) for f in files
                    if os.path.splitext(f)[1].lower() in EXTS]
    else:
        out = [os.path.join(directory, f) for f in os.listdir(directory)
               if os.path.splitext(f)[1].lower() in EXTS
               and os.path.isfile(os.path.join(directory, f))]
    return sorted(out)

def load_rgb(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")

def pad_box(box, w, h, pad):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (max(0, int(x0 - bw * pad)), max(0, int(y0 - bh * pad)),
            min(w, int(x1 + bw * pad)), min(h, int(y1 + bh * pad)))

def person_crops(img, conf, pad, min_size):
    w, h = img.size
    crops = []
    for box, _label, score in detect_person(img, conf_threshold=conf):
        b = pad_box(box, w, h, pad)
        if min(b[2] - b[0], b[3] - b[1]) < min_size:
            continue
        crops.append((b, float(score), img.crop(b)))
    if not crops:
        crops.append(((0, 0, w, h), 0.0, img))
    return crops

def head_crop(crop, pad):
    heads = detect_heads(crop)
    if not heads:
        return crop
    box, _label, _score = max(heads, key=lambda d: d[2])
    w, h = crop.size
    return crop.crop(pad_box(box, w, h, pad))

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
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--clean", action="store_true", help="Prune missing images from DB")
    ap.add_argument("--refresh", action="store_true", help="Purge existing tags for the character before grounding")
    
    args = ap.parse_args()

    ref_dir = os.path.abspath(args.reference_dir)
    # removed strict reference-dir path check to allow external reference directories

    db_path = os.path.join(base, f"{args.title}.sql")
    db = DBManager(db_path)

    if args.clean:
        print("Cleaning database...")
        confirm = input("WARNING: This will delete entries for images no longer on disk. Proceed? (y/N): ")
        if confirm.lower() == 'y':
            removed = db.prune_missing()
            print(f"Removed {removed} stale entries.")
        else:
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
        char_to_refresh = char_names[0] # Usually the only one in reference_dir
        print(f"Refreshing identity index for {char_to_refresh}...")
        removed = db.purge_character(char_to_refresh)
        print(f"Removed {removed} existing tags for {char_to_refresh}.")

    print("Updating identity index...")
    input_imgs = list_images(args.input, args.recursive)
    ref_imgs = list_images(ref_dir, recursive=False)
    
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
    results_path = os.path.join(args.output, "results.jsonl")
    counts = {"MATCH": 0, "DIST_FAIL": 0, "CLIP_FAIL": 0, "CLIP_ERROR": 0, "LOAD_ERROR": 0}
    clip_error_shown = False

    with open(results_path, "w") as log:
        for img_name, path in tqdm(all_db_images):
            rec = {"image": path, "matches": [], "crops": []}
            try:
                img = load_rgb(path)
                crops = person_crops(img, args.det_conf, args.pad, args.min_crop)
                crop_feats = [np.asarray(ccip_extract_feature(
                    head_crop(c[2], 0.25) if args.ccip_region == "head" else c[2]))
                    for c in crops]
                full = np.asarray(ccip_batch_differences(ref_feats + crop_feats))
                diffs = full[n_ref:, :n_ref]
            except Exception as e:
                rec["status"] = "LOAD_ERROR"
                rec["error"] = repr(e)
                counts["LOAD_ERROR"] += 1
                log.write(json.dumps(rec) + "\\n")
                continue

            best_per_char = {}
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
                log.write(json.dumps(rec) + "\\n")
                continue

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
                db.add_character_tag(img_name, name)
                
                rec["matches"].append(m)

            if any_match:
                rec["status"] = "MATCH"
            elif "CLIP_ERROR" in statuses:
                rec["status"] = "CLIP_ERROR"
            else:
                rec["status"] = "CLIP_FAIL"
            counts[rec["status"]] += 1
            log.write(json.dumps(rec) + "\\n")

    print("\\nDone.")
    for k, v in counts.items():
        print(f"  {k:<11} {v}")
    print(f"Log: {results_path}")
    db.close()

if __name__ == "__main__":
    main()
