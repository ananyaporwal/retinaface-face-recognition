"""LFW helpers: fix colour order / value range and avoid the stretched crop."""
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people, fetch_lfw_pairs

# sklearn's default crop is a tight 125x94 box; resizing that to 160x160 stretches
# faces ~33%. Use a roughly square crop (rows, cols of the 250x250 original).
# Heuristic: tune it by watching accuracy in tests/evaluate_recognition.py.
LFW_SLICE = (slice(60, 200), slice(55, 195))


def lfw_to_bgr(image):
    """sklearn LFW image (RGB, float 0-1) -> OpenCV image (BGR, uint8)."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        scale = 255.0 if image.max() <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def load_lfw_people(data_home, min_faces=6):
    return fetch_lfw_people(data_home=str(data_home), color=True, resize=1.0,
                            slice_=LFW_SLICE, min_faces_per_person=min_faces,
                            download_if_missing=True)


def load_lfw_pairs(data_home):
    return fetch_lfw_pairs(subset="10_folds", data_home=str(data_home), color=True,
                           resize=1.0, slice_=LFW_SLICE, download_if_missing=True)
