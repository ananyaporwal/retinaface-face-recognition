from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REGISTERED_FACES_DIR = DATA_DIR / "registered_faces"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pkl"
LFW_GALLERY_PATH = DATA_DIR / "lfw_gallery.pkl"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Cosine similarity in [-1, 1]; larger = more similar.
# 0.70 is only a guess. Run tests/evaluate_recognition.py and use the
# threshold it recommends, then re-check on your own webcam photos.
SIMILARITY_THRESHOLD = 0.41

# Optional: require the best match to beat the runner-up by this much.
# 0.0 disables it.
MATCH_MARGIN = 0.0

# Webcam device index. Change to 1 if camera 0 is unavailable.
CAMERA_INDEX = 0

# RetinaFace confidence filter (0 to 1).
DETECTION_THRESHOLD = 0.90

# Ignore faces smaller than this (pixels, shorter side). Tiny faces give
# unreliable embeddings.
MIN_FACE_SIZE = 40

# Align faces with RetinaFace landmarks before embedding.
ALIGN_FACES = True

# Webcam: run detection+embedding on every Nth frame (speeds up CPU use).
PROCESS_EVERY_N_FRAMES = 2

# --- LFW demo settings ---
LFW_MIN_FACES = 6          # only load people with at least this many images
LFW_HOLDOUT_PER_PERSON = 2 # images per person kept OUT of the gallery for testing
LFW_MAX_PEOPLE = 20
