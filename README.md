# CCIP Anime Face Grounding
This project implements a zero-shot identity grounding system for anime characters using the **CCIP (Contrastive Character Image Pretraining)** model via the `dghs-imgutils` library, with an optional **CLIP** attribute check to separate visually similar characters.

## Overview
The system finds a specific character in a collection of images by comparing them against a set of reference images.

For each input image:
1. Every person in the image is detected and cropped.
2. Each crop (or just its head, with `--ccip-region head`) is compared against every reference image using **CCIP's learned difference metric**.
3. The best-matching crop is kept if its difference is at or below the threshold.
4. Optionally, the matched crop is checked by **CLIP** against one or more yes/no prompt pairs (e.g. glasses vs. no glasses).
5. Matching images are copied to `output/Root_Ref/`, and all scores and bounding boxes are logged.

## Project Structure
- `ccip_grounder.py`: Main script for reference feature extraction and image grounding.
- `reference/`: Reference images of the target character. Only images in the root are used (subfolders are ignored); they are grouped as `Root_Ref`. Clean single-character headshots work best.
- `input/`: Images to search (full frames or crops).
- `output/Root_Ref/`: Matching images are copied here.
- `output/results.jsonl`: One line per input image with status, every detected crop's bounding box, CCIP difference, nearest reference image, and CLIP scores.
- `output/.ref_cache.pkl`: Cached reference features, so they are not re-extracted on every run. Rebuilt automatically when reference files or relevant options change.

## Setup and Usage
1. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
   - For a CPU-only PyTorch install (smaller download), first run: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Place reference images in the root of `reference/` and images to search in `input/`.
4. Run the grounder:

   Basic CCIP matching:
   ```
   python3 ccip_grounder.py --ccip-region head
   ```

   Recommended (CCIP + CLIP glasses check):
   ```
   python3 ccip_grounder.py --ccip-region head \
     --clip-pair "an anime girl wearing glasses|an anime girl not wearing glasses" \
     --clip-model openai/clip-vit-large-patch14-336
   ```

The CLIP model (about 1.7 GB) is downloaded on first use and cached locally.

## Options
| Option | Default | Description |
|---|---|---|
| `--reference` | `reference/` | Reference image directory (root only) |
| `--input` | `input/` | Input image directory |
| `--output` | `output/` | Output directory |
| `--recursive` | off | Also search subfolders of the **input** directory |
| `--threshold` | CCIP default (0.1785) | CCIP difference threshold; lower is stricter |
| `--topk` | 1 | Score = mean of the k closest references (1 = nearest neighbour) |
| `--crop-refs` | off | Crop the largest person out of each reference image (not needed for headshots) |
| `--ccip-region` | `person` | What CCIP compares: `person` crop or `head` only |
| `--det-conf` | 0.3 | Person detection confidence threshold |
| `--pad` | 0.1 | Padding around person boxes |
| `--min-crop` | 64 | Ignore detections smaller than this (px) |
| `--clip-pair` | none | `"YES prompt\|NO prompt"`; repeatable, all pairs must pass |
| `--clip-min` | 0.6 | Required P(YES) for each pair; use `0` to only log scores |
| `--clip-region` | `head` | Region CLIP checks: `head` or `person` |
| `--clip-model` | `openai/clip-vit-large-patch14` | Hugging Face CLIP model |
| `--save-crops` | off | Save the matched crop next to each copied image |

## Technical Notes
- **Identity model**: CCIP (ONNX, via `dghs-imgutils`).
- **Detection**: `imgutils` person and head detectors (ONNX).
- **Distance metric**: CCIP's own learned difference metric (`ccip_batch_differences`), not cosine distance. Lower means more similar.
- **Threshold**: `ccip_default_threshold()` (0.1785) unless overridden with `--threshold`.
- **Matching**: Nearest reference by default (`--topk 1`), not a mean anchor vector.
- **Attribute check**: CLIP via Hugging Face `transformers` (PyTorch). Each pair is scored as a softmax between its YES and NO prompts; a match must reach `--clip-min` on every pair. CLIP errors reject the image (fail closed) and are reported.
- **Calibration**: Run with `--clip-min 0` to log CLIP scores without filtering, then set `--clip-min` between the scores of true and false matches in `results.jsonl`.

## Milestones
- [x] **Milestone 1**: Basic grounding functionality implemented. (Matches confirmed, though distinctions between visually similar characters like Mai and Nano require further refinement/glasses detection).
- [x] **Milestone 1.5**: Head-region CCIP matching with headshot references, plus CLIP attribute verification (glasses check). Mai vs. Nano separation is now very good.
- [x] **Milestone 2**: Implement GPU acceleration for faster processing and iterative testing.
