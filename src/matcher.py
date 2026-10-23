"""Embedding storage and cosine-similarity matching."""
import pickle
from pathlib import Path
import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def save_embeddings(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(records, handle)


def load_embeddings(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Unexpected embedding file format: {path}")
    return records


def build_templates(records):
    """records [{'label','embedding'}] -> (names, (N,512) matrix), one row per person."""
    grouped = {}
    for r in records:
        grouped.setdefault(r["label"], []).append(
            np.asarray(r["embedding"], dtype=np.float32))
    names = sorted(grouped)
    rows = []
    for name in names:
        t = np.mean(grouped[name], axis=0)
        rows.append(t / (np.linalg.norm(t) + 1e-10))
    matrix = np.stack(rows) if rows else np.zeros((0, 512), np.float32)
    return names, matrix.astype(np.float32)


def match_templates(embedding, names, templates, threshold=0.70, margin=0.0):
    """Return (label, best_similarity). 'Unknown' if below threshold or ambiguous."""
    if len(names) == 0:
        return "Unknown", -1.0
    emb = np.asarray(embedding, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    sims = templates @ emb
    order = np.argsort(sims)[::-1]
    best = int(order[0])
    best_score = float(sims[best])
    if best_score < threshold:
        return "Unknown", best_score
    if margin > 0 and len(order) > 1 and best_score - float(sims[order[1]]) < margin:
        return "Unknown", best_score
    return names[best], best_score


def match_embedding(embedding, records, threshold=0.70):
    """Match against per-person average templates built from records."""
    names, templates = build_templates(records)
    return match_templates(embedding, names, templates, threshold)


# ---- LFW demo gallery (templates + held-out image indices) ----
def save_gallery(path, names, templates, heldout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"names": list(names), "templates": np.asarray(templates),
                     "heldout": heldout}, f)


def load_gallery(path):
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not (isinstance(data, dict) and "templates" in data):
        raise ValueError("Old gallery format. Re-run tests/create_gallery.py.")
    return data["names"], data["templates"], data["heldout"]
