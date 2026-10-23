"""Small WIDER FACE sample evaluation for RetinaFace (not the official protocol)."""
import random
import sys
from pathlib import Path
import cv2
from retinaface import RetinaFace

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "WIDER_val" / "images"
ANNOTATION_FILE = (DATA_DIR / "wider_face_split" / "wider_face_split"
                   / "wider_face_val_bbx_gt.txt")

MAX_IMAGES = 100
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.90
# Faces smaller than this (shorter side, px) are ignored in BOTH ground truth and
# predictions. In the first 100 val images ~80% of annotated faces are under 20 px,
# which no practical detector finds, so counting them makes recall meaningless.
MIN_FACE_SIZE = 30
SEED = 0


def calculate_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_annotations(path):
    """Parse WIDER annotations; drops 'invalid' faces. Handles 0-face entries."""
    lines = path.read_text(encoding="utf-8").splitlines()
    annotations, i = {}, 0
    while i < len(lines):
        name = lines[i].strip()
        i += 1
        if not name or i >= len(lines):
            continue
        try:
            count = int(lines[i].strip())
        except ValueError:
            continue
        i += 1
        boxes = []
        # An entry with 0 faces still has one placeholder line of zeros.
        for _ in range(max(count, 1)):
            if i >= len(lines):
                break
            values = lines[i].split()
            i += 1
            if count == 0 or len(values) < 4:
                continue
            x, y, w, h = map(int, values[:4])
            invalid = len(values) > 7 and int(values[7]) == 1
            if w > 0 and h > 0 and not invalid:
                boxes.append((x, y, w, h))
        annotations[name.replace("\\", "/")] = boxes
    return annotations


def main():
    if not IMAGE_DIR.exists():
        sys.exit(f"Image folder not found: {IMAGE_DIR}")
    if not ANNOTATION_FILE.exists():
        sys.exit(f"Annotation file not found: {ANNOTATION_FILE}")

    annotations = load_annotations(ANNOTATION_FILE)
    names = sorted(annotations)
    # Random sample across all 61 event folders (the first 100 are all "Parade").
    names = random.Random(SEED).sample(names, min(MAX_IMAGES, len(names)))

    total_gt = total_pred = total_matched = evaluated = 0
    for name in names:
        image = cv2.imread(str(IMAGE_DIR / name))
        if image is None:
            print(f"Skipping missing image: {IMAGE_DIR / name}")
            continue

        gts = [b for b in annotations[name] if min(b[2], b[3]) >= MIN_FACE_SIZE]
        total_gt += len(gts)

        result = RetinaFace.detect_faces(image, threshold=CONF_THRESHOLD)
        preds = []
        if isinstance(result, dict):
            for face in result.values():
                x1, y1, x2, y2 = map(int, face["facial_area"])
                if min(x2 - x1, y2 - y1) >= MIN_FACE_SIZE:
                    preds.append((x1, y1, x2 - x1, y2 - y1))
        total_pred += len(preds)

        cands = sorted(((calculate_iou(g, p), gi, pi)
                        for gi, g in enumerate(gts) for pi, p in enumerate(preds)
                        if calculate_iou(g, p) >= IOU_THRESHOLD), reverse=True)
        used_g, used_p = set(), set()
        for _, gi, pi in cands:
            if gi not in used_g and pi not in used_p:
                used_g.add(gi)
                used_p.add(pi)
        total_matched += len(used_g)
        evaluated += 1
        if evaluated % 10 == 0:
            print(f"Processed {evaluated}/{len(names)} images")

    if evaluated == 0:
        sys.exit("No images were evaluated. Check the dataset paths.")
    precision = total_matched / total_pred if total_pred else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    print("\n--- WIDER FACE Sample Evaluation ---")
    print(f"Images evaluated: {evaluated} (random sample, seed {SEED})")
    print(f"Faces >= {MIN_FACE_SIZE}px, valid: {total_gt}")
    print(f"Predicted faces >= {MIN_FACE_SIZE}px: {total_pred}")
    print(f"Matched (IoU >= {IOU_THRESHOLD}): {total_matched}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nNot the official WIDER FACE benchmark (no Easy/Medium/Hard split).")


if __name__ == "__main__":
    main()
