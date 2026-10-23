"""FaceNet-style embeddings. ONE preprocessing path shared by every script."""
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

INPUT_SIZE = 160

# Standard 5-point face template (112x112), scaled to 160x160.
# Order: left eye, right eye, nose, left mouth corner, right mouth corner
# (left/right as seen in the image).
_TEMPLATE_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
TEMPLATE = _TEMPLATE_112 * (INPUT_SIZE / 112.0)


def pad_to_square(image):
    """Pad (never stretch) an image to a square so faces keep their shape."""
    h, w = image.shape[:2]
    if h == w:
        return image
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(image, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def square_crop(image_bgr, box, margin=0.2):
    """Square crop around a box with some margin; pads if it leaves the frame."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (1.0 + margin) / 2.0
    nx1, ny1 = int(round(cx - half)), int(round(cy - half))
    nx2, ny2 = int(round(cx + half)), int(round(cy + half))
    pl, pt = max(0, -nx1), max(0, -ny1)
    pr, pb = max(0, nx2 - w), max(0, ny2 - h)
    if pl or pt or pr or pb:
        image_bgr = cv2.copyMakeBorder(image_bgr, pt, pb, pl, pr,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        nx1, nx2, ny1, ny2 = nx1 + pl, nx2 + pl, ny1 + pt, ny2 + pt
    return image_bgr[ny1:ny2, nx1:nx2]


def align_face(image_bgr, landmarks):
    """Warp the face to the 160x160 template using 5 landmarks. None if unusable."""
    try:
        eyes = sorted([landmarks["right_eye"], landmarks["left_eye"]],
                      key=lambda p: p[0])
        mouth = sorted([landmarks["mouth_right"], landmarks["mouth_left"]],
                       key=lambda p: p[0])
        pts = np.array([eyes[0], eyes[1], landmarks["nose"], mouth[0], mouth[1]],
                       dtype=np.float32)
    except (KeyError, TypeError, IndexError):
        return None
    matrix, _ = cv2.estimateAffinePartial2D(pts, TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(image_bgr, matrix, (INPUT_SIZE, INPUT_SIZE),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class FaceEmbedder:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Downloads pretrained weights the first time it is initialized.
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @staticmethod
    def _prepare(face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Received an empty face crop.")
        face = pad_to_square(face_bgr)
        if face.shape[0] != INPUT_SIZE:
            interp = cv2.INTER_AREA if face.shape[0] > INPUT_SIZE else cv2.INTER_CUBIC
            face = cv2.resize(face, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)
        return face

    @torch.inference_mode()
    def embed_batch(self, faces_bgr, batch_size=64):
        """List of BGR uint8 face images (any size) -> (N, 512) L2-normalised."""
        prepared = [self._prepare(f) for f in faces_bgr]
        outputs = []
        for start in range(0, len(prepared), batch_size):
            chunk = np.stack(prepared[start:start + batch_size])[..., ::-1]  # BGR->RGB
            x = torch.from_numpy(np.ascontiguousarray(chunk)).float()
            x = x.permute(0, 3, 1, 2)
            x = (x - 127.5) / 128.0            # facenet-pytorch standardisation
            emb = self.model(x.to(self.device)).cpu().numpy().astype(np.float32)
            outputs.append(emb)
        emb = np.concatenate(outputs) if outputs else np.zeros((0, 512), np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-10)

    def get_embedding(self, face_bgr):
        """Embedding for a single BGR face crop."""
        return self.embed_batch([face_bgr])[0]

    def embed_detection(self, image_bgr, face, align=True):
        """Embedding for one detection dict from detect_faces()."""
        crop = None
        if align:
            crop = align_face(image_bgr, face.get("landmarks"))
        if crop is None:
            crop = square_crop(image_bgr, face["box"])
        return self.get_embedding(crop)
