"""RetinaFace detection wrapper."""
from retinaface import RetinaFace


def detect_faces(frame_bgr, confidence_threshold=0.90, min_size=0):
    """Return detections: dicts with 'box' (x1,y1,x2,y2), 'score', 'landmarks'.

    frame_bgr must be a BGR uint8 image (what cv2.imread / VideoCapture give).
    """
    # Pass the threshold INTO RetinaFace. Previously the library always used
    # its own 0.9 default, so values below 0.9 had no effect.
    raw = RetinaFace.detect_faces(frame_bgr, threshold=confidence_threshold)
    if not isinstance(raw, dict):
        return []

    height, width = frame_bgr.shape[:2]
    results = []
    for item in raw.values():
        area = item.get("facial_area")
        if not area or len(area) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in area]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        score = float(item.get("score", 0.0))
        if x2 <= x1 or y2 <= y1 or score < confidence_threshold:
            continue
        if min(x2 - x1, y2 - y1) < min_size:
            continue
        landmarks = {k: (float(v[0]), float(v[1]))
                     for k, v in (item.get("landmarks") or {}).items()}
        results.append({"box": (x1, y1, x2, y2), "score": score,
                        "landmarks": landmarks})
    return results
