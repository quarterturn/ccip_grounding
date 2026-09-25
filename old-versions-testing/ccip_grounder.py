import os
import json
import shutil
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from imgutils.metrics import ccip_extract_feature, ccip_default_threshold
from imgutils.generic.clip import CLIPModel

# --- CONFIGURATION ---
REFERENCE_DIR = "/home/alex/Documents/ccip_grounding/reference"
INPUT_DIR = "/home/alex/Documents/ccip_grounding/input"
OUTPUT_DIR = "/home/alex/Documents/ccip_grounding/output"
ANCHORS_FILE = "/home/alex/Documents/ccip_grounding/anchors.json"
DEBUG_FILE = "/home/alex/Documents/ccip_grounding/debug_best_matches.txt"

try:
    THRESHOLD = ccip_default_threshold()
except Exception:
    THRESHOLD = 0.6

def get_image_paths(directory):
    if not os.path.exists(directory): return []
    valid_exts = (".png", ".jpg", ".jpeg", ".webp")
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_exts)]

def get_cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    sim = np.dot(a / norm_a, b / norm_b)
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
                feat = ccip_extract_feature(img)
                features.append(feat.tolist() if hasattr(feat, 'tolist') else feat)
            except Exception as e:
                print(f"Error extracting feature from {img}: {e}")
        if features:
            anchors["Root_Ref"] = features 
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
    parser = argparse.ArgumentParser(description="CCIP Character Grounder with CLIP Filtering")
    parser.add_argument('--positive-clip', type=str, default=None, help="Positive CLIP prompt")
    parser.add_argument('--negative-clip', type=str, default=None, help="Negative CLIP prompt")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anchors = build_anchors()
    if not anchors: return

    input_imgs = get_image_paths(INPUT_DIR)
    if not input_imgs:
        print(f"No images found in {INPUT_DIR}!")
        return

    print(f"Using Distance Threshold: {THRESHOLD}")
    print(f"Grounding {len(input_imgs)} images using Nearest Neighbor Cosine Distance...")
    
    clip_model = None
    target_model_name = None
    if args.positive_clip or args.negative_clip:
        print(f"CLIP Filtering Active: Positive='{args.positive_clip}', Negative='{args.negative_clip}'")
        try:
            # Use Xenova repo which is the standard for ONNX-based CLIP in many JS/Py environments
            repo_id = 'Xenova/clip-vit-base-patch32'
            clip_model = CLIPModel(repo_id)
            
            # Robust model name detection
            if clip_model.model_names:
                target_model_name = clip_model.model_names[0]
            else:
                # In many cases, the model is in the root or under a specific folder
                # We'll try common names.
                candidates = ['clip-vit-base-patch32', '']
                found = False
                for c in candidates:
                    try:
                        clip_model._check_model_name(c)
                        target_model_name = c
                        found = True
                        break
                    except:
                        continue
                if not found:
                    target_model_name = ''
            
            print(f"Loaded CLIP model: {repo_id} (Variant: '{target_model_name}')")
        except Exception as e:
            print(f"CRITICAL: Failed to load CLIP model: {e}")
            clip_model = None

    match_count = 0
    debug_results = []

    for img_path in tqdm(input_imgs):
        try:
            feat = np.array(ccip_extract_feature(img_path))
            best_char = None
            min_dist = float("inf")

            for char_name, anchor_feats in anchors.items():
                current_min_dist = min([get_cosine_distance(feat, ref_feat) for ref_feat in anchor_feats])
                if current_min_dist < min_dist:
                    min_dist = current_min_dist
                    best_char = char_name
            
            passed_clip = True
            pos_score = 0.0
            neg_score = 0.0
            
            if clip_model:
                pos = args.positive_clip if args.positive_clip else "a person"
                neg = args.negative_clip if args.negative_clip else "no person"
                
                try:
                    res = clip_model.predict(
                        images=[img_path], 
                        texts=[pos, neg], 
                        model_name=target_model_name
                    )
                    preds = res['predictions'] if isinstance(res, dict) else res
                    probs = preds[0]
                    pos_score = probs[0]
                    neg_score = probs[1]
                    if pos_score < neg_score:
                        passed_clip = False
                except Exception as e:
                    pos_score = -1.0 
                    neg_score = -1.0
                    passed_clip = True

            if min_dist <= THRESHOLD and passed_clip:
                dest_dir = os.path.join(OUTPUT_DIR, best_char)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(img_path, dest_dir)
                match_count += 1
            
            status = "MATCH" if (min_dist <= THRESHOLD and passed_clip) else ("CLIP_FAIL" if not passed_clip else "DIST_FAIL")
            debug_results.append(f"{os.path.basename(img_path)} | {best_char} | {min_dist:.4f} | {pos_score:.4f} | {neg_score:.4f} | {status}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    with open(DEBUG_FILE, "w") as f:
        f.write("Image | Best Char | CCIP Dist | CLIP Pos | CLIP Neg | Status\n" + "-"*70 + "\n")
        f.write("\n".join(debug_results))

    print(f"\nGrounding complete! Matches found: {match_count}")
    print(f"Debug log saved to: {DEBUG_FILE}")

if __name__ == "__main__":
    main()
