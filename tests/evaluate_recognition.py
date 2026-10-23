"""LFW verification (same person / different person) with proper folds."""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import DATA_DIR
from src.embedder import FaceEmbedder
from src.lfw_utils import lfw_to_bgr, load_lfw_pairs


def best_threshold(scores, labels):
    grid = np.linspace(-0.2, 1.0, 241)
    acc = [accuracy_score(labels, scores >= t) for t in grid]
    return float(grid[int(np.argmax(acc))])


def main():
    print("Loading LFW pairs...")
    lfw = load_lfw_pairs(DATA_DIR)
    embedder = FaceEmbedder()

    n = len(lfw.pairs)
    scores = np.zeros(n, dtype=np.float32)
    step = 64
    for start in range(0, n, step):
        chunk = lfw.pairs[start:start + step]
        a = embedder.embed_batch([lfw_to_bgr(p[0]) for p in chunk])
        b = embedder.embed_batch([lfw_to_bgr(p[1]) for p in chunk])
        scores[start:start + step] = np.sum(a * b, axis=1)
        if (start // step) % 10 == 0:
            print(f"Processed {min(start + step, n)}/{n} pairs")

    y = lfw.target.astype(int)
    # The official LFW folds are 10 CONSECUTIVE blocks (600 pairs each).
    # (The old `index % 10` split mixed pairs across folds.)
    fold_size = n // 10
    folds = np.arange(n) // fold_size
    preds = np.zeros_like(y)
    thresholds, fold_acc = [], []
    for k in range(10):
        test = folds == k
        t = best_threshold(scores[~test], y[~test])   # tuned on the other 9 folds
        preds[test] = (scores[test] >= t).astype(int)
        thresholds.append(t)
        fold_acc.append(accuracy_score(y[test], preds[test]))

    print("\n--- LFW Recognition Evaluation (10-fold) ---")
    print(f"Pairs: {n}")
    print(f"Accuracy: {np.mean(fold_acc):.4f} +/- {np.std(fold_acc):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y, scores):.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y, preds))
    print(f"Mean same-person score:      {scores[y == 1].mean():.3f}")
    print(f"Mean different-person score: {scores[y == 0].mean():.3f}")
    print(f"\nRecommended SIMILARITY_THRESHOLD: {np.mean(thresholds):.2f}  "
          f"(fold range {min(thresholds):.2f}-{max(thresholds):.2f})")
    print("Put this value in config.py. Not an official FaceNet benchmark.")


if __name__ == "__main__":
    main()
