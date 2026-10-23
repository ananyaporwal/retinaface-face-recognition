"""Run detection + recognition on one WIDER image and save the annotated result.

Usage: python tests/visualize_detection.py [path/to/image.jpg]
Faces from WIDER are random strangers, so mostly 'Unknown' is the CORRECT result
against a 20-person LFW gallery.
"""
import sys
from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import (ALIGN_FACES, DATA_DIR, DETECTION_THRESHOLD, FIGURES_DIR,
                    LFW_GALLERY_PATH, MATCH_MARGIN, MIN_FACE_SIZE,
                    SIMILARITY_THRESHOLD)
from src.detector import detect_faces
from src.embedder import FaceEmbedder
from src.matcher import load_gallery, match_templates

IMAGE_DIR = DATA_DIR / "WIDER_val" / "images"


def main():
    names, templates, _ = load_gallery(LFW_GALLERY_PATH)
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        paths = sorted(IMAGE_DIR.rglob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No JPG images found in {IMAGE_DIR}")
        image_path = paths[0]
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read {image_path}")

    embedder = FaceEmbedder()
    faces = detect_faces(frame, DETECTION_THRESHOLD, MIN_FACE_SIZE)
    for face in faces:
        emb = embedder.embed_detection(frame, face, align=ALIGN_FACES)
        label, score = match_templates(emb, names, templates,
                                       SIMILARITY_THRESHOLD, MATCH_MARGIN)
        x1, y1, x2, y2 = face["box"]
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({score:.2f})", (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "recognition_result.jpg"
    cv2.imwrite(str(out), frame)
    print("Input image:", image_path)
    print(f"Faces processed (>= {MIN_FACE_SIZE}px): {len(faces)}")
    print("Output saved to:", out)


if __name__ == "__main__":
    main()
