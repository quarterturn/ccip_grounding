# CCIP Anime Face Grounding

This project implements a zero-shot identity grounding system for anime characters using the **CCIP (Contrastive Character Image Pretraining)** model via the `dghs-imgutils` library, with an optional **CLIP** attribute check to separate visually similar characters.

## Overview
The system finds a specific character in a collection of images by comparing them against a set of reference images.

For each input image:
1. Every person in the image is detected and cropped.
2. Each crop (or just its head, with `--ccip-region head`) is compared against every reference image using **CCIP's learned difference metric**.
3. The best-matching crop is kept if its difference is at or below the threshold.
4. Optionally, the matched crop is checked by **CLIP** against one or more yes/no prompt pairs (e.g. glasses vs. no glasses).
5. Matches are logged to a persistent SQLite database and results are written to `output/`.

## Project Structure
- `ccip_grounder.py`: The main grounding script with persistent SQLite tracking.
- `ground_all.py`: Orchestrator to process all characters in the `reference/` directory.
- `audit_viewer.py`: Tool for auditing grounding results via contact sheets.
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
| `--save-crops` | No | Save the matched crop to output |
| `--clean` | No | Prune images no longer on disk from the database |
| `--refresh` | No | Purge existing tags for the character before grounding |

## Technical Notes
- **Identity model**: CCIP (ONNX). Distance metric is a learned difference, not cosine distance.
- **Persistence**: Uses SQLite to avoid redundant scans and allow multi-character identity layering.
- **Tagging**: The character tag is derived automatically from the `--reference-dir` folder name.

## Milestones
- [x] **Milestone 1**: Basic grounding functionality implemented.
- [x] **Milestone 1.5**: Head-region matching and CLIP attribute verification.
- [x] **Milestone 2**: GPU acceleration.
- [x] **Milestone 3**: Persistent SQLite identity indexing and automated character tagging.
