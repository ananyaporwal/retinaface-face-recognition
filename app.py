"""Real-time webcam face recognition demo."""
import cv2

from config import (
    ALIGN_FACES, CAMERA_INDEX, DETECTION_THRESHOLD, EMBEDDINGS_PATH,
    MATCH_MARGIN, MIN_FACE_SIZE, PROCESS_EVERY_N_FRAMES, SIMILARITY_THRESHOLD,
)
from src.detector import detect_faces
from src.embedder import FaceEmbedder
from src.matcher import build_templates, load_embeddings, match_templates


def main():
    records = load_embeddings(EMBEDDINGS_PATH)
    if not records:
        print("No registered embeddings found.")
        print("1. Add photos under data/registered_faces/<PersonName>/")
        print("2. Run: python register_faces.py")
        return
    names, templates = build_templates(records)
    print(f"Loaded {len(records)} embeddings for {len(names)} people: {names}")

    print("Loading embedding model (first run may download weights)...")
    embedder = FaceEmbedder()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open webcam index {CAMERA_INDEX}.")
        print("Check camera permissions or change CAMERA_INDEX in config.py.")
        return

    print("Webcam started. Press 'q' in the video window to quit.")
    frame_no, last_results, error_text = 0, [], ""
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            if frame_no % PROCESS_EVERY_N_FRAMES == 0:
                try:
                    last_results, error_text = [], ""
                    for face in detect_faces(frame, DETECTION_THRESHOLD, MIN_FACE_SIZE):
                        emb = embedder.embed_detection(frame, face, align=ALIGN_FACES)
                        label, sim = match_templates(
                            emb, names, templates, SIMILARITY_THRESHOLD, MATCH_MARGIN)
                        last_results.append((face["box"], label, sim))
                except Exception as exc:  # keep the window alive
                    error_text = f"Detection error: {str(exc)[:60]}"
            frame_no += 1

            for (x1, y1, x2, y2), label, sim in last_results:
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({sim:.2f})", (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            if error_text:
                cv2.putText(frame, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("RetinaFace + FaceNet Recognition (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
