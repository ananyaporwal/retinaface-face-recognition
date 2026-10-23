"""Held-out recognition demo: one known person and one person NOT in the gallery."""
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import (DATA_DIR, FIGURES_DIR, LFW_GALLERY_PATH, LFW_MIN_FACES,
                    SIMILARITY_THRESHOLD)
from src.embedder import FaceEmbedder
from src.lfw_utils import lfw_to_bgr, load_lfw_people
from src.matcher import load_gallery, match_templates


def annotate(bgr, title, label, score, truth):
    img = cv2.resize(bgr, (400, 400), interpolation=cv2.INTER_CUBIC)
    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (399, 399), color, 3)
    for i, text in enumerate([f"True: {truth}", f"{title}: {label}", f"similarity {score:.2f}"]):
        cv2.putText(img, text, (8, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return img


def main():
    names, templates, heldout = load_gallery(LFW_GALLERY_PATH)
    dataset = load_lfw_people(DATA_DIR, LFW_MIN_FACES)
    embedder = FaceEmbedder()

    known = names[0]
    known_idx = heldout[known][0]
    outsider_id = next(i for i, n in enumerate(dataset.target_names) if n not in names)
    outsider = dataset.target_names[outsider_id]
    outsider_idx = int(np.where(dataset.target == outsider_id)[0][0])

    tiles = []
    for title, idx, truth in [("Held-out known", known_idx, known),
                              ("Not in gallery", outsider_idx, outsider)]:
        bgr = lfw_to_bgr(dataset.images[idx])
        emb = embedder.get_embedding(bgr)
        label, score = match_templates(emb, names, templates, SIMILARITY_THRESHOLD)
        print(f"{truth:25s} -> {label} ({score:.3f})")
        tiles.append(annotate(bgr, title, label, score, truth))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "lfw_recognition_demo.jpg"
    cv2.imwrite(str(out), np.hstack(tiles))
    print("Saved:", out)
    print("Expected: first = correct name, second = Unknown. Two images, not a benchmark.")


if __name__ == "__main__":
    main()
