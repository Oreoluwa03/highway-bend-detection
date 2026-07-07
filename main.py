import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import time
import os
import requests
from PIL import Image
from io import BytesIO
from collections import deque

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Highway Bend Detection API",
    description="""
## CNN-Based Highway Bend & Corner Detection

Detects whether a road image shows a **sharp bend** or a **straight road**
using a fine-tuned MobileNetV2 model.

### Endpoints
- `POST /predict` — Upload a road image, get back a prediction
- `GET /health`   — Check if the API and model are loaded
- `GET /`         — API info
""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ──────────────────────────────────────────────────────────────────────
CLASSES = ["sharp", "straight"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pth")
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def build_model():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2)
    )
    model = model.to(device)
    model.eval()
    return model


# Download model from Google Drive if not present
GDRIVE_ID = "1KTJhLGultbBeRpN59XLpaI4x3VnjOFGb"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        import gdown
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_ID}",
            MODEL_PATH,
            quiet=False
        )
        print("✅ Model downloaded successfully")
download_model()

# Load model at startup
model_loaded = False

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model = build_model()
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model = build_model()
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            model = build_model()
            model.load_state_dict(checkpoint, strict=False)
    else:
        model = checkpoint
    model.eval()
    model_loaded = True
    print(f"✅ Model loaded from {MODEL_PATH}")
else:
    model = build_model()
    print(f"⚠️  model.pth not found — running with random weights")
    # ── Response schemas ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    is_bend: bool
    is_alert: bool
    processing_time_ms: float
    all_scores: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    classes: list
# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "name": "Highway Bend Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        device=str(device),
        classes=CLASSES
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="Road image (jpg, png, webp)")):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp)")

    # Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Try a different file.")

    # Preprocess & predict
    start = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    elapsed_ms = (time.time() - start) * 1000

    is_bend = (pred_class == 0)
    is_alert = is_bend and confidence > 0.7

    return PredictionResponse(
        prediction=CLASSES[pred_class],
        confidence=round(confidence * 100, 2),
        is_bend=is_bend,
        is_alert=is_alert,
        processing_time_ms=round(elapsed_ms, 2),
        all_scores={
            "sharp": round(probs[0].item() * 100, 2),
            "straight": round(probs[1].item() * 100, 2)
        }
    )
