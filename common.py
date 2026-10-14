"""Shared image utilities for CCIP grounding tools.

Single source of truth for file listing, RGB loading, and person/head
cropping so the grounder and the dataset tools always agree on how a
crop is produced (feature cache keys depend on it).
"""
import os

from PIL import Image

from imgutils.detect import detect_person, detect_heads

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(directory, recursive):
    if not os.path.isdir(directory):
        return []
    out = []
    if recursive:
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
    crops = []
    for box, _label, score in detect_person(img, conf_threshold=conf):
        b = pad_box(box, *img.size, pad)
        if min(b[2] - b[0], b[3] - b[1]) < min_size:
            continue
        crops.append((b, float(score), img.crop(b)))
    if not crops:
        crops.append(((0, 0, *img.size), 0.0, img))
    return crops


def head_crop(crop, pad):
    heads = detect_heads(crop)
    if not heads:
        return crop
    box, _label, _score = max(heads, key=lambda d: d[2])
    return crop.crop(pad_box(box, *crop.size, pad))
