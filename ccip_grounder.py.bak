import os
import json
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from imgutils.metrics import ccip_extract_feature, ccip_default_threshold

# --- CONFIGURATION ---
REFERENCE_DIR = "/home/alex/Documents/ccip_grounding/reference"
INPUT_DIR = "/home/alex/Documents/ccip_grounding/input"
OUTPUT_DIR = "/home/alex/Documents/ccip_grounding/output"
ANCHORS_FILE = "/home/alex/Documents/ccip_grounding/anchors.json"
DEBUG_FILE = "/home/alex/Documents/ccip_grounding/debug_best_matches.txt"

# CCIP default threshold is usually a distance. 
# We will use a permissive threshold for now to ensure we get matches.
try:
    THRESHOLD = ccip_default_threshold()
except Exception:
    THRESHOLD = 0.6

print(f"Using Distance Threshold: {THRESHOLD}")

def get_image_paths(directory):
    if not os.path.exists(directory): return []
    valid_exts = (".png", ".jpg", ".jpeg", ".webp")
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_exts)]

def get_cosine_distance(a, b):
    # Ensure vectors are numpy arrays
    a = np.array(a)
    b = np.array(b)
    
    # Normalize vectors to unit length
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 1.0
    
    # Normalize and dot product = cosine similarity
    sim = np.dot(a / norm_a, b / norm_b)
    # Distance = 1 - similarity
    return 1.0 - sim

def build_anchors():
    print(f"Building anchors from {REFERENCE_DIR} (Root Only)...")
    anchors = {}
    
    root_imgs = get_image_paths(REFERENCE_DIR)
    if root_imgs:
        print(f"  -> Found {len(root_imgs)} images in root. Creating anchor Root_Ref")
        features = []
        for img in root_imgs:
            try:
                features.append(ccip_extract_feature(img))
            except Exception as e:
                print(f"Error extracting feature from {img}: {e}")
        
        if features:
            anchors["Root_Ref"] = np.mean(features, axis=0).tolist()
        else:
            print("CRITICAL: No features could be extracted from reference images!")
            return None
    else:
        print("CRITICAL: No reference images found in the root of the reference folder!")
        return None

    with open(ANCHORS_FILE, "w") as f:
        json.dump(anchors, f)
    print(f"Success! Created anchors for {len(anchors)} characters.\n")
    return anchors

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anchors = build_anchors()
    if not anchors: return

    input_imgs = get_image_paths(INPUT_DIR)
    if not input_imgs:
        print(f"No images found in {INPUT_DIR}!")
        return

    print(f"Grounding {len(input_imgs)} images using Normalized Cosine Distance...")
    match_count = 0
    debug_results = []

    for img_path in tqdm(input_imgs):
        try:
            feat = np.array(ccip_extract_feature(img_path))
            best_char = None
            min_dist = float("inf")

            for char_name, anchor_feat in anchors.items():
                dist = get_cosine_distance(feat, anchor_feat)
                if dist < min_dist:
                    min_dist = dist
                    best_char = char_name
            
            debug_results.append(f"{os.path.basename(img_path)} | Best: {best_char} | Dist: {min_dist:.4f}")

            # Use the threshold for the normalized distance
            if min_dist <= THRESHOLD:
                dest_dir = os.path.join(OUTPUT_DIR, best_char)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(img_path, dest_dir)
                match_count += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    with open(DEBUG_FILE, "w") as f:
        f.write("Image | Best Match | Cosine Distance\n" + "-"*40 + "\n")
        f.write("\n".join(debug_results))

    print(f"\nGrounding complete! Matches found: {match_count}")
    print(f"Debug log saved to: {DEBUG_FILE}")

if __name__ == "__main__":
    main()
