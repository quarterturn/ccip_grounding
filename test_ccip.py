from PIL import Image
import os
import random
from imgutils.metrics import ccip_extract_feature

dataset_path = os.path.expanduser("~/Documents/image-training-datasets/anime-faces-v1")
# Find all image files
images = [os.path.join(root, f) for root, _, files in os.walk(dataset_path) for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if not images:
    print("No images found in dataset path!")
    exit(1)

test_img_path = random.choice(images)
print(f"Testing with image: {test_img_path}")

try:
    img = Image.open(test_img_path).convert("RGB")
    feat = ccip_extract_feature(img)
    print(f"Success! Feature vector shape: {feat.shape}")
    print(f"Vector sample: {feat[:5]}...")
except Exception as e:
    print(f"Error: {e}")
