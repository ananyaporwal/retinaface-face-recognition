"""Create local face embeddings from data/registered_faces/<person>/*.jpg."""
import cv2

from config import (ALIGN_FACES, DETECTION_THRESHOLD, EMBEDDINGS_PATH,
                    MIN_FACE_SIZE, REGISTERED_FACES_DIR)
from src.detector import detect_faces
from src.embedder import FaceEmbedder
from src.matcher import save_embeddings

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    REGISTERED_FACES_DIR.mkdir(parents=True, exist_ok=True)
    person_dirs = [p for p in REGISTERED_FACES_DIR.iterdir() if p.is_dir()]
    if not person_dirs:
        print(f"No person folders found in: {REGISTERED_FACES_DIR}")
        print("Create data/registered_faces/PersonName/ and add clear face photos.")
        return

    embedder = FaceEmbedder()
    records, per_person = [], {}
    for person_dir in sorted(person_dirs):
        images = [p for p in person_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            print(f"[SKIP] No supported images in {person_dir.name}")
            continue
        for image_path in sorted(images):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[SKIP] Could not read {image_path.name}")
                continue
            try:
                faces = detect_faces(image, DETECTION_THRESHOLD, MIN_FACE_SIZE)
                if not faces:
                    print(f"[SKIP] No face detected in {person_dir.name}/{image_path.name}")
                    continue
                # Several faces: use the largest one.
                face = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) *
                                                (f["box"][3] - f["box"][1]))
                emb = embedder.embed_detection(image, face, align=ALIGN_FACES)
                records.append({"label": person_dir.name, "embedding": emb})
                per_person[person_dir.name] = per_person.get(person_dir.name, 0) + 1
                print(f"[OK] Registered {person_dir.name}/{image_path.name}")
            except Exception as exc:
                print(f"[ERROR] {image_path}: {exc}")

    if not records:
        print("No embeddings were created. Check the photos and try again.")
        return
    save_embeddings(EMBEDDINGS_PATH, records)
    print(f"\nSaved {len(records)} embeddings to: {EMBEDDINGS_PATH}")
    for name, n in per_person.items():
        if n < 3:
            print(f"[TIP] {name} has only {n} usable photo(s); 3-5 varied photos work better.")
    print("Keep this file and the original face photos private.")


if __name__ == "__main__":
    main()
