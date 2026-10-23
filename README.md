# RetinaFace + FaceNet Face Recognition

A starter project for real-time face detection and recognition using:
- **RetinaFace** for face detection
- **FaceNet-style embeddings** (`InceptionResnetV1`, pretrained on VGGFace2) for face recognition
- **OpenCV** for webcam input and display

> This is a learning/portfolio starter, not a security or access-control system. Recognition can be wrong. Use only with informed consent, and keep face images and embeddings private.

## 1. Setup (Windows / VS Code)

Open this project folder in VS Code. In the terminal, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal and retry activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The first run downloads pretrained model weights and may take a few minutes. Internet access is required for the initial download.

## 2. Register faces

Create a folder for each consenting person, then place a few clear face photos in it:

```text
data/
  registered_faces/
    Alice/
      photo1.jpg
      photo2.jpg
    Bob/
      photo1.jpg
```

You can use `.jpg`, `.jpeg`, `.png`, and `.webp` images. Keep each image focused on one face.

Then run:
```powershell
python register_faces.py
```

This creates `data/embeddings.pkl`, a local file containing face embeddings and labels. Do not upload this file or the photos to GitHub.

## 3. Run the webcam app

```powershell
python app.py
```

- Press **q** to quit.
- Faces not close enough to a registered embedding are labelled **Unknown**.
- If your webcam is not camera 0, edit `CAMERA_INDEX` in `config.py`.

## 4. Tune recognition

`SIMILARITY_THRESHOLD` in `config.py` is a starting point, not a universal value. Higher thresholds are stricter and may label more faces Unknown; lower thresholds may cause false matches. Evaluate on images/videos that are separate from the registration photos before drawing conclusions.

## Project structure

```text
retinaface-face-recognition/
├── app.py
├── register_faces.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── registered_faces/   # Add your own consented photos
│   ├── samples/
│   └── embeddings.pkl      # Generated locally
├── src/
│   ├── detector.py
│   ├── embedder.py
│   └── matcher.py
└── tests/
    └── test_matcher.py
```

## Notes and limitations

- RetinaFace is used for detection; FaceNet-style embeddings are used for recognition. They are separate pretrained components.
- This starter uses a face crop resized to the embedding model's input size. It does not yet perform landmark-based face alignment or a formal accuracy evaluation.
- Lighting, pose, camera quality, and threshold choice affect results.
- For a portfolio-ready version, add landmark-based alignment, a held-out evaluation set, precision/recall or ROC analysis, and clear limitations in the report.
