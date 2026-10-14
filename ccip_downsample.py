#!/usr/bin/env python3
"""
CCIP Diversity Downsampler.
Reduces a large set of grounded images to a balanced, diverse subset per character.
Uses Max-Min distance sampling to ensure maximum visual variety.
"""
import argparse
import sqlite3
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imgutils.detect import detect_person, detect_heads
from imgutils.metrics import ccip_extract_feature, ccip_default_threshold
from dbutil import open_v2, distinct_chars

def load_rgb(path):
    from PIL import Image
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")

def get_best_crop(img, region='head'):
    # Use the same cropping logic as the grounder for consistency
    if region == 'head':
        heads = detect_heads(img)
        if not heads: return img
        box, _, _ = max(heads, key=lambda d: d[2])
        w, h = img.size
        # Simple pad
        bw, bh = box[2]-box[0], box[3]-box[1]
        x0, y0, x1, y1 = box
        return img.crop((max(0, int(x0-bw*0.25)), max(0, int(y0-bh*0.25)), 
                        min(w, int(x1+bw*0.25)), min(h, int(y1+bh*0.25))))
    return img

def extract_feature(path, region='head'):
    try:
        img = load_rgb(path)
        crop = get_best_crop(img, region)
        return np.asarray(ccip_extract_feature(crop))
    except Exception as e:
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

def main():
    parser = argparse.ArgumentParser(description="Downsample grounded images for diversity.")
    parser.add_argument("--db", required=True, help="Source .sql database")
    parser.add_argument("--out-db", required=True, help="Output .sql database")
    parser.add_argument("--target", type=int, default=100, help="Target images per character")
    parser.add_argument("--region", choices=['head', 'person'], default='head', help="Region to use for diversity calc")
    args = parser.parse_args()

    conn = open_v2(args.db)
    conn.row_factory = sqlite3.Row

    chars = distinct_chars(conn)
    print(f"Found {len(chars)} characters to balance.")

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
            "SELECT i.path, i.name FROM images i "
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

        # 2. Diversity Sampling
        selected_indices = max_min_sample(feats, args.target)
        
        # 3. Save to new DB (carry ALL tags of a selected image, not just
        #    the character being balanced, to preserve multi-character rows)
        for idx in selected_indices:
            row = valid_rows[idx]
            tags = [t[0] for t in conn.execute(
                "SELECT char FROM image_tags WHERE path = ?", (row['path'],))]
            out_conn.execute(
                "INSERT OR REPLACE INTO images (path, name) VALUES (?, ?)",
                (row['path'], row['name']))
            out_conn.execute("DELETE FROM image_tags WHERE path = ?",
                             (row['path'],))
            for t in tags:
                out_conn.execute(
                    "INSERT OR IGNORE INTO image_tags (path, char) VALUES (?, ?)",
                    (row['path'], t))
        
        out_conn.commit()

    conn.close()
    out_conn.close()
    print(f"Balanced database created at: {args.out_db}")

if __name__ == "__main__":
    main()
