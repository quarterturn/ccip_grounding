# CCIP Anime Face Grounding

This project implements a 0-shot identity grounding system for anime characters using the **CCIP (Contrastive Character Image Pre-training)** model via the `dghs-imgutils` library.

## Overview
The system identifies specific characters in a large collection of face crops (extracted via YOLO) by comparing them against a set of reference images. It calculates the mean feature vector (anchor) for each character and uses **Normalized Cosine Distance** to determine matches.

## Project Structure
- `ccip_grounder.py`: Main script for anchor generation and image grounding.
- `reference/`: Directory containing reference images. Images in the root are grouped as `Root_Ref`.
- `input/`: Directory containing face crops to be grounded.
- `output/`: Directory where matching images are copied into character-specific folders.
- `anchors.json`: Saved identity vectors to avoid re-calculating anchors.
- `debug_best_matches.txt`: Log of the closest match and distance for every processed image.

## Setup and Usage
1. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Place reference images in `reference/` and targets in `input/`.
4. Run the grounder: `python3 ccip_grounder.py`

## Technical Notes
- **Model**: CCIP
- **Distance Metric**: Normalized Cosine Distance (1 - Cosine Similarity).
- **Threshold**: Uses `ccip_default_threshold()` for matching.

## Milestones
- [x] **Milestone 1**: Basic grounding functionality implemented. (Matches confirmed, though distinctions between visually similar characters like Mai and Nano require further refinement/glasses detection).
- [ ] **Milestone 2**: Implement GPU acceleration for faster processing and iterative testing.
