# CCIP Anime Face Grounding

This project implements a zero-shot identity grounding system for anime characters using the **CCIP (Contrastive Character Image Pretraining)** model via the `dghs-imgutils` library, with an optional **CLIP** attribute check to separate visually similar characters.

## Overview
The system finds a specific character in a collection of images by comparing them against a set of reference images.

For each input image:
1. Every person in the image is detected and cropped.
2. Each crop (or just its head, with `--ccip-region head`) is compared against every reference image using **CCIP's learned difference metric**. Crops are extracted in batches (`--batch-size`) for throughput.
3. The best-matching crop is kept if its difference is at or below the threshold.
4. Optionally, the matched crop is checked by **CLIP** against one or more yes/no prompt pairs (e.g. glasses vs. no glasses).
5. Matches are logged to a persistent SQLite database and results are written to `output/` (one `results_<character>.jsonl` per run).

## Project Structure
- `ccip_grounder.py`: The main grounding script with persistent SQLite tracking.
- `ground_all.py`: Orchestrator to process all characters in the `reference/` directory.
- `migrate_v2.py`: One-time migration of legacy (v1) databases to the v2 schema.
- `dbutil.py`: Shared v2-schema DB helpers (refuses legacy databases).
- `audit_viewer.py`: Tool for auditing grounding results via contact sheets.
- `create_audit_sheets.py`: Batch contact sheets for every character in a database.
- `ccip_downsample.py`: Max-min diversity downsampling per character.
- `ccip_hard_cap.py`: Random pruning to enforce a strict per-character cap.
- `db_prune.py`: Remove database entries whose files no longer exist on disk.
- `gate_parity.py`: A/B the facesaver YOLO gate (AGPL) vs the MIT deepghs
  face detector — gate-decision agreement tables for the product licensing swap.
- `reference/`: Reference images of target characters. Use subdirectories for each character (e.g. `reference/minakami_mai/`).
- `input/`: Images to search.
- `output/`: Matching results, logs, and reference caches.
- `(title).sql`: SQLite database storing the identity index for a specific show.

## Setup and Usage
1. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Place reference images in character-specific subfolders under `reference/`.
4. Run the grounder:

### Persistent Identity Indexing
The grounder uses a database to track characters across multiple runs. You can process one character at a time to build a comprehensive index for a show.

```bash
python3 ccip_grounder.py --title "nichijou" --reference-dir "reference/minakami_mai" --ccip-region head --clip-pair "an anime girl wearing glasses|an anime girl not wearing glasses"
```

Then run it for another character:
```bash
python3 ccip_grounder.py --title "nichijou" --reference-dir "reference/naganohara_mio" --ccip-region head --clip-pair "an anime girl with blue hair|an anime girl without blue hair"
```

The `nichijou.sql` database will now track which images contain Mai, Mio, or both.

## Options for `ccip_grounder.py`
| Option | Required | Description |
|---|---|---|
| `--title` | Yes | Title of the show (used as the database filename) |
| `--reference-dir` | Yes | Path to the character's reference folder (must be under `reference/`) |
| `--input` | No | Input image directory (default: `input/`) |
| `--output` | No | Output directory (default: `output/`) |
| `--recursive` | No | Search subfolders of the **input** directory |
| `--threshold` | No | CCIP diff threshold; lower is stricter |
| `--topk` | No | Score = mean of k closest refs (default: 1) |
| `--crop-refs` | No | Crop the largest person out of reference images |
| `--ccip-region` | No | `person` or `head` (default: `person`) |
| `--det-conf` | No | Person detection confidence (default: 0.3) |
| `--pad` | No | Padding around person boxes (default: 0.1) |
| `--min-crop` | No | Ignore detections smaller than this (default: 64px) |
| `--clip-pair` | No | `"YES prompt\|NO prompt"`; repeatable |
| `--clip-min` | No | Required P(YES) for each pair (default: 0.6) |
| `--clip-region` | No | `head` or `person` (default: `head`) |
| `--clip-model` | No | HF CLIP model path |
| `--clean` | No | Prune images no longer on disk from the database |
| `--yes` | No | Skip the `--clean` confirmation prompt (for scripted runs) |
| `--refresh` | No | Purge existing tags for the character before grounding |
| `--gpu` | No | Pin ONNX + CLIP to a physical GPU ID (e.g. `--gpu 1`) |
| `--batch-size` | No | Images loaded/detected per batch (default: 32) |
| `--max-crops` | No | Max crops per batched CCIP ONNX call (default: 8; bounds VRAM) |

## Technical Notes
- **Identity model**: CCIP (ONNX). Distance metric is a learned difference, not cosine distance.
- **Batched extraction**: `--batch-size` images are loaded/detected per pass; CCIP ONNX calls then run in chunks of `--max-crops` crops (bounds activation memory — 83 crops at 384px OOMed ~8GB on a 3090). Measured on .80 (48 stills, RTX 3090, ~100 refs): CCIP is compute-saturated, so chunk sizes 8–32 all land within noise of per-crop timing; 8 measured fastest and is the default. The real win is robustness: a failed chunk retries per-crop instead of killing the run.
- **Batched vs per-image numerics**: ONNX float accumulation order varies with batch shape. Measured end-to-end drift: max ~6e-05 in CCIP distance, zero status flips across test sets. If you ever set thresholds with margins smaller than ~1e-4, run with `--batch-size 1` for reproducibility.
- **Persistence**: Uses SQLite to avoid redundant scans and allow multi-character identity layering.
- **Schema (v2)**: `images(path PRIMARY KEY, name)` plus an `image_tags(path, char)` join table. Paths are the key (no basename collisions) and tags are exact-match (no substring bleed). Legacy v1 databases are detected and rejected; migrate them with `python3 migrate_v2.py --db <file> --dry-run` first, then without `--dry-run`. The original v1 table is preserved as `images_legacy`.
- **Tagging**: The character tag is derived automatically from the `--reference-dir` folder name.

## Gate parity harness (`gate_parity.py`) — product detector swap

The product build cannot ship Ultralytics + `yolov8x6_animeface.pt`
(AGPL-3.0). This harness measures whether the MIT-licensed
`deepghs/anime_face_detection` model (`face_detect_v1.4_s` via imgutils
ONNX — the license family the grounding crop stage already uses) makes the
same frame gate decisions as facesaver. Agreement is measured on gate
*decisions*: both detectors' boxes pass through facesaver's exact geometry
gate (min 100px, area 0.5–30%, aspect 0.45–1.6), and both run at the
production defaults (YOLO conf 0.3, imgsz 1280). DeepGHS runs once at the
lowest sweep threshold and higher thresholds are re-gated from recorded
scores — exact, because NMS never lets a low-score box suppress a
higher-score one. Card-recommended threshold for v1.4_s is 0.307; do NOT
assume detector confidences transfer — read the sweep.

```bash
# Keep side: the stills that passed the historical YOLO gate (grounding DB).
python3 gate_parity.py --db madeinabyss.sql --gpu 1

# Both sides: raw frames that never met a gate (uniform stride sampling:
# ffmpeg -vf "select=not(mod(n\,125))" -vsync vfr -q:v 3 .../%05d.jpg).
python3 gate_parity.py --images raw_frames --gpu 1
```

Outputs in `<input>_gate_parity/`: `decisions.csv`, `summary.txt`
(agreement/recall per threshold), `evidence/{yolo_only,dg_only}/` copies
of the strongest disagreements, `*_all.txt` full lists. Pin `--gpu 1`
before any CUDA-using import; the script warms the CUDA context before
importing ultralytics (it rewrites `CUDA_VISIBLE_DEVICES` on first
predict() otherwise and silently lands on GPU0).

## Milestones
- [x] **Milestone 1**: Basic grounding functionality implemented.
- [x] **Milestone 1.5**: Head-region matching and CLIP attribute verification.
- [x] **Milestone 2**: GPU acceleration.
- [x] **Milestone 3**: Persistent SQLite identity indexing and automated character tagging.
- [x] **Milestone 4**: v2 path-keyed schema, batched pipeline, pure-logic test suite.

## Testing
Pure-logic tests (no GPU, models, or datasets required; needs numpy):

```bash
python3 -m unittest discover -s tests -v
```

They cover scoring (topk mean-of-k selection), CLIP pair parsing, max-min
diversity sampling, the v2 DB layer (path keying, tag semantics, legacy
rejection), the migration script, and the hard-cap pruner.
