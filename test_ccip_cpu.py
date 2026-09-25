import os
import random
import numpy as np
from PIL import Image
from imgutils.metrics import ccip_extract_feature

# Try to force CPU by overriding providers if possible, 
# though imgutils usually handles this internally.
# We will check if it runs without GPU acceleration first.

dataset_path = os.path.expanduser("~/Documents/image-training-datasets/anime-faces-v1")
images = [os.path.join(root, f) for root, _, files in os.walk(dataset_path) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not images:
    print("No images found in dataset path!")
    exit(1)

test_img_path = random.choice(images)
print(f"Testing (CPU focus) with image: {test_img_path}")

try:
    img = Image.open(test_img_path).convert("RGB")
    # We pass the model explicitly to see if we can trigger a different load path
    feat = ccip_extract_feature(img, model="ccip-caformer-24-randaug-pruned")
    print(f"Success! Feature vector shape: {feat.shape}")
    print(f"Vector sample: {feat[:5]}...")
except Exception as e:
    print(f"Error: {e}")
