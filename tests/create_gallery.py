"""Build a small LFW gallery. Held-out images are NOT used in the templates."""
import sys
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import (DATA_DIR, LFW_GALLERY_PATH, LFW_HOLDOUT_PER_PERSON,
                    LFW_MAX_PEOPLE, LFW_MIN_FACES)
from src.embedder import FaceEmbedder
from src.lfw_utils import lfw_to_bgr, load_lfw_people
from src.matcher import save_gallery


def main():
    print("Loading LFW dataset (first run with this crop re-reads the images)...")
    dataset = load_lfw_people(DATA_DIR, LFW_MIN_FACES)
    embedder = FaceEmbedder()

    counts = np.bincount(dataset.target)
    # Pick the people with the most images (more stable templates).
    chosen = np.argsort(counts)[::-1][:LFW_MAX_PEOPLE]

    names, templates, heldout = [], [], {}
    for pid in chosen:
        name = dataset.target_names[pid]
        idx = np.where(dataset.target == pid)[0]
        test_idx = idx[-LFW_HOLDOUT_PER_PERSON:]
        train_idx = idx[:-LFW_HOLDOUT_PER_PERSON]
        emb = embedder.embed_batch([lfw_to_bgr(dataset.images[i]) for i in train_idx])
        t = emb.mean(axis=0)
        t /= np.linalg.norm(t) + 1e-10
        names.append(name)
        templates.append(t.astype(np.float32))
        heldout[name] = [int(i) for i in test_idx]
        print(f"Added {name}: {len(train_idx)} gallery images, {len(test_idx)} held out")

    save_gallery(LFW_GALLERY_PATH, names, np.stack(templates), heldout)
    print(f"\nGallery with {len(names)} identities saved to: {LFW_GALLERY_PATH}")
    print("Demonstration only, not secure identification.")


if __name__ == "__main__":
    main()
