# RetinaFace Face Recognition

A face detection and recognition project using RetinaFace, FaceNet embeddings, and cosine similarity.

## Features
- Face detection using RetinaFace
- Face embeddings using FaceNet
- Similarity-based identity matching
- Unknown-face rejection
- Detection and recognition evaluation scripts

## Tech Stack
- Python
- OpenCV
- RetinaFace
- FaceNet (InceptionResnetV1)
- NumPy
- PyTorch

## Project Structure
- `app.py` — webcam recognition application
- `register_faces.py` — register identities from face images
- `src/` — detection, embeddings, and matching modules
- `tests/` — evaluation and demo scripts
- `reports/figures/` — sample output images

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

For the webcam application, register face images and generate the required embeddings first.

## Evaluation

The project includes scripts for LFW recognition evaluation and sampled WIDER FACE detection evaluation. Results are sample evaluations, not official benchmark results.

## Privacy

Downloaded datasets, local face images, embeddings, and model weights should not be committed to this repository.

## Status

This project is under development. More improvements and documentation will be added.