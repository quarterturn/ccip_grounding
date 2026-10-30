#!/usr/bin/env python3
"""
CCIP Diversity Downsampler.
Reduces a large set of grounded images to a balanced, diverse subset per character.
Uses Max-Min distance sampling to ensure maximum visual variety.
"""
import argparse
import sqlite3

import numpy as np
from tqdm import tqdm
from imgutils.metrics import ccip_extract_feature
from common import load_rgb, head_crop
from dbutil import open_v2, distinct_chars

def extract_feature(path, region='head'):
    try:
        img = load_rgb(path)
        crop = head_crop(img, 0.25) if region == 'head' else img
        return np.asarray(ccip_extract_feature(crop))
    except Exception:
        return None

def max_min_sample(features, target_count):
    if len(features) <= target_count:
        return list(range(len(features)))
    
    # Start with the first image
    selected_indices = [0]
    # Distances from each point to its closest selected point
    min_distances = np.linalg.norm(features - features[0], axis=1)
    
    while len(selected_indices) < target_count:
        # Pick the point that is furthest from its nearest selected neighbor
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
        
        # Update min_distances: distance to the new member of the selected set
        new_distances = np.linalg.norm(features - features[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        
    return selected_indices

def select_prefer_solo(feats, n_tags, target):
    """Selection plan for --prefer-solo: solo-tagged images first, group shots
    backfill, each pool Max-Min-sampled on its own so diversity pressure stays
    inside both. Pure function (features in, positional indices out) so the
    pool-split logic is unit-testable without GPU or models.
    """
    n_tags = np.asarray(n_tags)
    solo_idx = np.where(n_tags == 1)[0]
    group_idx = np.where(n_tags > 1)[0]

    selected = []
    for pool_idx in (solo_idx, group_idx):
        room = target - len(selected)
        if room <= 0:
            break
        if len(pool_idx) == 0:
            continue
        if len(pool_idx) <= room:
            selected.extend(pool_idx.tolist())
        else:
            picks = max_min_sample(feats[pool_idx], room)
            selected.extend(pool_idx[picks].tolist())
    return selected


def main():
    parser = argparse.ArgumentParser(description="Downsample grounded images for diversity.")
    parser.add_argument("--db", required=True, help="Source .sql database")
    parser.add_argument("--out-db", required=True, help="Output .sql database")
    parser.add_argument("--target", type=int, default=100, help="Target images per character")
    parser.add_argument("--region", choices=['head', 'person'], default='head', help="Region to use for diversity calc")
    parser.add_argument("--prefer-solo", action="store_true",
                        help="Fill the target from solo-tagged images first, "
                             "backfilling from multi-character group shots only "
                             "when the solo pool runs dry (better identity "
                             "binding for LoRA training).")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Physical GPU index to pin to (sets CUDA_VISIBLE_DEVICES before any model loads).")
    parser.add_argument("--ov-gpu", action="store_true",
                        help="Run ONNX models on the Intel iGPU via OpenVINO "
                             "device_type=GPU (chibi; loud CPU fallback).")
    args = parser.parse_args()

    if args.ov_gpu:
        import ovpatch
        ovpatch.enable_if_requested(cli_flag=True)

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    conn = open_v2(args.db)
    conn.row_factory = sqlite3.Row

    chars = distinct_chars(conn)
    print(f"Found {len(chars)} characters to balance.")

    tag_accum = {}  # path -> full tag list (carried across characters)
    names = {}      # path -> name
    counts = {}     # char -> selected count

    # Setup output DB (v2 schema)
    out_conn = sqlite3.connect(args.out_db)
    out_conn.execute("CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, name TEXT)")
    out_conn.execute("CREATE TABLE IF NOT EXISTS image_tags ("
                     "path TEXT NOT NULL, char TEXT NOT NULL, "
                     "PRIMARY KEY (path, char))")
    out_conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_char ON image_tags(char)")

    for char in tqdm(chars, desc="Balancing Characters"):
        # Exact-match join: no substring bleed between similar tag names
        rows = conn.execute(
            "SELECT i.path, i.name, "
            "(SELECT COUNT(*) FROM image_tags t2 WHERE t2.path = i.path) AS n_tags "
            "FROM images i "
            "JOIN image_tags t ON i.path = t.path WHERE t.char = ?",
            (char,)).fetchall()
        
        if not rows: continue

        # 1. Extract features for all candidates
        feats = []
        valid_rows = []
        for row in rows:
            f = extract_feature(row['path'], args.region)
            if f is not None:
                feats.append(f)
                valid_rows.append(row)
        
        if not feats: continue
        feats = np.array(feats)
        n_tags = np.array([r['n_tags'] for r in valid_rows])

        # 2. Diversity Sampling. With --prefer-solo, run Max-Min independently
        #    on the solo pool and the group pool, then concatenate solo-first up
        #    to the target. Sampling each pool on its own keeps Max-Min's
        #    diversity pressure *within* the group shots too, and chibi/reaction
        #    frames (extreme face proportions) sit far from normal close-ups in
        #    CCIP space, so they get picked as diverse members either way.
        if args.prefer_solo:
            selected_indices = select_prefer_solo(feats, n_tags, args.target)
        else:
            selected_indices = max_min_sample(feats, args.target)
        
        # 3. Save to new DB (carry ALL tags of a selected image, not just
        #    the character being balanced, to preserve multi-character rows).
        #    Tags accumulate in memory across characters; rewriting each
        #    image's tag set at the end avoids wiping tags another character
        #    already committed for shared group shots.
        for idx in selected_indices:
            row = valid_rows[idx]
            if row['path'] not in tag_accum:
                tag_accum[row['path']] = [t[0] for t in conn.execute(
                    "SELECT char FROM image_tags WHERE path = ?", (row['path'],))]
                names[row['path']] = row['name']
        
        counts[char] = len(selected_indices)

    for path, tags in tag_accum.items():
        out_conn.execute(
            "INSERT OR REPLACE INTO images (path, name) VALUES (?, ?)",
            (path, names[path]))
        for t in tags:
            out_conn.execute(
                "INSERT OR IGNORE INTO image_tags (path, char) VALUES (?, ?)",
                (path, t))
    out_conn.commit()

    conn.close()
    out_conn.close()
    print(f"Balanced database created at: {args.out_db}")
    for char in sorted(counts, key=counts.get, reverse=True):
        uniq = sum(1 for p, t in tag_accum.items() if char in t)
        print(f"  {char:12s} selected={counts[char]:4d}  unique-tagged-in-out={uniq}")

if __name__ == "__main__":
    main()
