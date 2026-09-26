# CCIP Anime Face Grounding
# (mirroring to github)

This project implements a zero-shot identity grounding system for anime characters using the **CCIP (Contrastive Character Image Pretraining)** model via the `dghs-imgutils` library, with an optional **CLIP** attribute check to separate visually similar characters.

## Overview
The system finds a specific character in a collection of images by comparing them against a set of reference images.

For each input image:
1. Every person in the image is detected and cropped.
2. Each crop (or just its head, with `--ccip-region head`) is compared against every reference image using **CCIP's learned difference metric**.
3. The best-matching crop is kept if its difference is at or below the threshold.
4. Optionally, the matched crop is checked by **CLIP** against one or more yes/no prompt pairs (e.g. glasses vs. no glasses).
5. Matching images are copied to `output/` and all scores and bounding boxes are logged.

## Project Structure
- `ccip_grounder.py`: Original basic grounding script.
- `ccip_grounder2.py`: Enhanced version with persistent SQLite tracking.
- `reference/`: Reference images of target characters. Use subdirectories for each character (e.g. `reference/minakami_mai/`).
- `input/`: Images to search.
- `output/`: Matching images grouped by character name.
- `(title).sql`: SQLite database storing the identity index for a specific show.
- `output/results.jsonl`: Match logs and bounding boxes.
- `output/.ref_cache.pkl`: Cached reference features.

## Setup and Usage
1. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Place reference images in character-specific subfolders under `reference/` (e.g. `reference/minakami_mai/`).
4. Run the grounder:

### Persistent Identity Indexing (`ccip_grounder2.py`)
This version allows you to run the tool multiple times for different characters in the same show, progressively building a comprehensive identity index in a database.

```bash
python3 ccip_grounder2.py --title "nichijou" --reference-dir "reference/minakami_mai" --ccip-region head --clip-pair "an anime girl wearing glasses|an anime girl not wearing glasses"
```

Then run it for another character:
```bash
python3 ccip_grounder2.py --title "nichijou" --reference-dir "reference/naganohara_mio" --ccip-region head --clip-pair "an anime girl with blue hair|an anime girl without blue hair"
```

The `nichijou.sql` database will now track which images contain Mai, Mio, or both.

## Options for `ccip_grounder2.py`
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
| `--clip-pair` | No | `\"YES prompt\\|NO prompt\"`; repeatable |
| `--clip-min` | No | Required P(YES) for each pair (default: 0.6) |
| `--clip-region` | No | `head` or `person` (default: `head`) |
| `--clip-model` | No | HF CLIP model path |
| `--save-crops` | No | Save the matched crop to output |
| `--clean` | No | Prune images no longer on disk from the database |

## Technical Notes
- **Identity model**: CCIP (ONNX). Distance metric is a learned difference, not cosine distance.
- **Persistence**: Uses SQLite to avoid redundant scans and allow multi-character identity layering.
- **Tagging**: The character tag is derived automatically from the `--reference-dir` folder name.

## Milestones
- [x] **Milestone 1**: Basic grounding functionality implemented.
- [x] **Milestone 1.5**: Head-region matching and CLIP attribute verification.
- [x] **Milestone 2**: GPU acceleration.
- [x] **Milestone 3**: Persistent SQLite identity indexing and automated character tagging.
