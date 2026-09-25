#!/usr/bin/env python3
"""
CCIP character grounding with optional CLIP attribute checks.

Reference images: placed directly in reference/ (subfolders are ignored).
All reference images are treated as one character, "Root_Ref".

Pipeline per input image:
  1. Detect every person -> crop each one
  2. CCIP difference of each crop vs. every reference (learned metric, NOT cosine)
  3. Assign each crop to its closest character; keep it if difference <= threshold
  4. Optional CLIP yes/no attribute check on that crop (head or person region)
  5. Copy image to output/<character>/ and log boxes + scores to results.jsonl

Install: pip install dghs-imgutils transformers torch pillow tqdm numpy
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import traceback

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


# ---------------------------------------------------------------- utilities
def list_images(directory, recursive):
    if not os.path.isdir(directory):
        return []
    out = []
    if recursive:
        #for root, _, files in os.walk(directory):
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
    """Return list of (box, det_score, crop). Falls back to the whole image."""
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
    """Best head inside a person crop; returns the person crop if none found."""
    heads = detect_heads(crop)
    if not heads:
        return crop
    box, _label, _score = max(heads, key=lambda d: d[2])
    w, h = crop.size
    return crop.crop(pad_box(box, w, h, pad))


# ---------------------------------------------------------------- references
def build_anchors(ref_dir, cache_path, crop_refs, det_conf, pad):
    root_imgs = list_images(ref_dir, recursive=False)
    if not root_imgs:
        sys.exit(f"CRITICAL: no reference images found in the root of {ref_dir}")
    groups = {"Root_Ref": root_imgs}
    if os.path.isdir(ref_dir):
        for name in sorted(os.listdir(ref_dir)):
            sub = os.path.join(ref_dir, name)
            if os.path.isdir(sub):
                imgs = list_images(sub, recursive=True)
                if imgs:
                    groups[name] = imgs
    if not groups:
        sys.exit(f"CRITICAL: no reference images found in {ref_dir}")

    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    new_cache, anchors = {}, {}
    for name, paths in groups.items():
        feats = []
        for p in tqdm(paths, desc=f"ref [{name}]"):
            key = (os.path.abspath(p), os.path.getmtime(p),
                   os.path.getsize(p), crop_refs)
            if key in cache:
                feat = cache[key]
            else:
                img = load_rgb(p)
                if crop_refs:  # use the largest detected person
                    crops = person_crops(img, det_conf, pad, 0)
                    img = max(crops, key=lambda c: c[2].size[0] * c[2].size[1])[2]
                feat = np.asarray(ccip_extract_feature(img))
            new_cache[key] = feat
            feats.append(feat)
        anchors[name] = feats
        print(f"  {name}: {len(feats)} reference features")

    with open(cache_path, "wb") as f:
        pickle.dump(new_cache, f)
    return anchors


# ---------------------------------------------------------------- CLIP
class ClipChecker:
    """Each pair is (yes_prompt, no_prompt). Score = softmax P(yes) within the pair."""

    def __init__(self, model_name, pairs, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.pairs = pairs

        texts = [t for pair in pairs for t in pair]
        with torch.no_grad():
            tok = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
            tf = self.model.get_text_features(**tok)
        self.text = tf / tf.norm(dim=-1, keepdim=True)
        self.scale = self.model.logit_scale.exp()

    def score(self, crop):
        torch = self.torch
        with torch.no_grad():
            px = self.proc(images=crop, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**px)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (self.scale * f @ self.text.T)[0]
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
    ap = argparse.ArgumentParser(description="CCIP character grounder + CLIP attribute check")
    base = "/home/alex/Documents/ccip_grounding"
    ap.add_argument("--reference", default=f"{base}/reference")
    ap.add_argument("--input", default=f"{base}/input")
    ap.add_argument("--output", default=f"{base}/output")
    ap.add_argument("--recursive", action="store_true", help="search input subfolders")
    ap.add_argument("--threshold", type=float, default=None,
                    help="CCIP difference threshold (lower = stricter). Default: CCIP's own.")
    ap.add_argument("--topk", type=int, default=1,
                    help="score = mean of the k closest references (1 = nearest neighbour)")
    ap.add_argument("--crop-refs", action="store_true",
                    help="crop the largest person out of each reference image")
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--pad", type=float, default=0.1, help="padding around person boxes")
    ap.add_argument("--min-crop", type=int, default=64, help="ignore tiny detections (px)")
    ap.add_argument("--clip-pair", action="append",
                    help='repeatable. e.g. "an anime girl wearing glasses|an anime girl with no glasses"')
    ap.add_argument("--clip-min", type=float, default=0.6,
                    help="required P(yes) for every pair")
    ap.add_argument("--clip-region", choices=["head", "person"], default="head")
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    ap.add_argument("--save-crops", action="store_true",
                    help="also save the matched crop next to each copied image")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    threshold = args.threshold if args.threshold is not None else float(ccip_default_threshold())
    print(f"CCIP difference threshold: {threshold:.4f}")

    anchors = build_anchors(args.reference, os.path.join(args.output, ".ref_cache.pkl"),
                            args.crop_refs, args.det_conf, args.pad)
    char_names = list(anchors.keys())
    ref_feats, ref_owner = [], []
    for ci, name in enumerate(char_names):
        ref_feats += anchors[name]
        ref_owner += [ci] * len(anchors[name])
    ref_owner = np.array(ref_owner)
    n_ref = len(ref_feats)

    pairs = parse_pairs(args.clip_pair)
    clip = None
    if pairs:
        print(f"Loading CLIP: {args.clip_model}")
        clip = ClipChecker(args.clip_model, pairs)  # crash here if it fails to load
        for yes, no in pairs:
            print(f"  require P('{yes}') >= {args.clip_min}  vs  '{no}'")

    inputs = list_images(args.input, args.recursive)
    if not inputs:
        sys.exit(f"No images found in {args.input}")
    print(f"Processing {len(inputs)} images")

    results_path = os.path.join(args.output, "results.jsonl")
    counts = {"MATCH": 0, "DIST_FAIL": 0, "CLIP_FAIL": 0, "CLIP_ERROR": 0, "LOAD_ERROR": 0}
    clip_error_shown = False

    with open(results_path, "w") as log:
        for path in tqdm(inputs):
            rec = {"image": path, "matches": [], "crops": []}
            try:
                img = load_rgb(path)
                crops = person_crops(img, args.det_conf, args.pad, args.min_crop)
                crop_feats = [np.asarray(ccip_extract_feature(c[2])) for c in crops]
                # learned CCIP metric; take the crops x references block
                full = np.asarray(ccip_batch_differences(ref_feats + crop_feats))
                diffs = full[n_ref:, :n_ref]
            except Exception as e:
                rec["status"] = "LOAD_ERROR"
                rec["error"] = repr(e)
                counts["LOAD_ERROR"] += 1
                log.write(json.dumps(rec) + "\n")
                continue

            # per crop: score for each character = mean of its k closest refs
            best_per_char = {}  # char index -> (score, crop index)
            for k, (box, det_score, _crop) in enumerate(crops):
                char_scores = []
                for ci in range(len(char_names)):
                    d = np.sort(diffs[k, ref_owner == ci])
                    char_scores.append(float(d[:max(1, min(args.topk, len(d)))].mean()))
                ci_best = int(np.argmin(char_scores))
                s_best = char_scores[ci_best]
                rec["crops"].append({"box": list(box), "det_score": det_score,
                                     "best_char": char_names[ci_best], "ccip_diff": s_best})
                if s_best <= threshold:
                    if ci_best not in best_per_char or s_best < best_per_char[ci_best][0]:
                        best_per_char[ci_best] = (s_best, k)

            if not best_per_char:
                rec["status"] = "DIST_FAIL"
                counts["DIST_FAIL"] += 1
                log.write(json.dumps(rec) + "\n")
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
                        m["status"] = "CLIP_ERROR"  # fail closed
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
                dest = os.path.join(args.output, name)
                os.makedirs(dest, exist_ok=True)
                flat = os.path.relpath(path, args.input).replace(os.sep, "__")
                shutil.copy2(path, os.path.join(dest, flat))
                if args.save_crops:
                    stem = os.path.splitext(flat)[0]
                    crop.save(os.path.join(dest, f"{stem}.crop{k}.png"))
                rec["matches"].append(m)

            if any_match:
                rec["status"] = "MATCH"
            elif "CLIP_ERROR" in statuses:
                rec["status"] = "CLIP_ERROR"
            else:
                rec["status"] = "CLIP_FAIL"
            counts[rec["status"]] += 1
            log.write(json.dumps(rec) + "\n")

    print("\nDone.")
    for k, v in counts.items():
        print(f"  {k:<11} {v}")
    print(f"Log: {results_path}")
    if counts["CLIP_ERROR"]:
        print("WARNING: CLIP errors occurred - those images were rejected, see traceback above.")


if __name__ == "__main__":
    main()
