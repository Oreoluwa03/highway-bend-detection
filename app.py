
import streamlit as st
import requests
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sharp Bend & Corner Detector",
    page_icon="🛣️",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .bend  { background: #fff1f0; border-left: 5px solid #ff4d4f; color: #a8071a; }
    .safe  { background: #f6ffed; border-left: 5px solid #52c41a; color: #135200; }
    .alert { background: #fff0f0; border-left: 5px solid #cf1322; color: #820014; }
</style>
""", unsafe_allow_html=True)

# ── API URL ────────────────────────────────────────────────────────────────────
API_URL = "https://highway-bend-api.onrender.com"

# ── Load model for live feed (runs locally, no API call needed) ────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pth")
CLASSES = ["sharp", "straight"]
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    m = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in m.parameters():
        param.requires_grad = False
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(m.last_channel, 128),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2)
    )
    m = m.to(device)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict):
            try:
                m.load_state_dict(checkpoint)
            except RuntimeError:
                m.load_state_dict(checkpoint, strict=False)
        else:
            m = checkpoint
    m.eval()
    return m

# ── Video processor for live feed ─────────────────────────────────────────────
class BendDetectorProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.history = deque(maxlen=3)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()

        self.history.append(pred)
        smoothed = max(set(self.history), key=self.history.count)
        is_bend = smoothed == 0
        is_alert = is_bend and conf > 0.7

        # Draw overlay
        h, w = img.shape[:2]
        if is_alert:
            cv2.rectangle(img, (0, 0), (w, 70), (0, 0, 200), -1)
            cv2.putText(img, "⚠ SHARP BEND!", (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        elif is_bend:
            cv2.rectangle(img, (0, 0), (w, 55), (0, 0, 180), -1)
            cv2.putText(img, "BEND DETECTED", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (0, 0), (w, 55), (0, 140, 0), -1)
            cv2.putText(img, "STRAIGHT ROAD", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(img, f"{conf*100:.0f}%", (w - 90, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🛣️ Sharp Bend & Corner Detector")
st.markdown("CNN-based road bend detection using **MobileNetV2** — upload an image or use your live camera.")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Info")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("API: Online ✅")
        st.write(f"**Model loaded:** {'Yes ✅' if health['model_loaded'] else 'No ❌'}")
        st.write(f"**Device:** {health['device']}")
    except Exception:
        st.warning("API: Offline ⚠️")
    st.divider()
    st.markdown("**Model:** MobileNetV2")
    st.markdown("**Classes:** sharp, straight")
    st.markdown("**Threshold:** 70% confidence")
    st.markdown("[📖 API Docs](http://localhost:8000/docs)")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Live Camera"])

# ── Tab 1: Image Upload ────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Clear front-facing road photos work best"
    )

    if uploaded:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded, caption="Uploaded image", use_column_width=True)
        with col2:
            with st.spinner("Analysing road..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data["is_alert"]:
                            st.markdown('<div class="result-box alert">⚠️ SHARP BEND ALERT</div>', unsafe_allow_html=True)
                        elif data["is_bend"]:
                            st.markdown('<div class="result-box bend">⚠️ Sharp Bend Detected</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-box safe">✅ Straight Road</div>', unsafe_allow_html=True)
                        st.metric("Confidence", f"{data['confidence']:.1f}%")
                        st.metric("Processing time", f"{data['processing_time_ms']:.0f} ms")
                        scores = data["all_scores"]
                        st.progress(scores["sharp"] / 100, text=f"Sharp: {scores['sharp']:.1f}%")
                        st.progress(scores["straight"] / 100, text=f"Straight: {scores['straight']:.1f}%")
                        with st.expander("📦 Raw API response"):
                            st.json(data)
                    else:
                        st.error(f"API error {resp.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("FastAPI backend is offline. Start it with: uvicorn backend.main:app --reload")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload a road image to get started")

# ── Tab 2: Live Camera ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("Point your camera at a road — the model will detect bends in real time.")
    st.warning("Allow camera access when prompted by your browser.")

    RTC_CONFIG = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    webrtc_streamer(
        key="bend-detector",
        video_processor_factory=BendDetectorProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Final Year Project — CNN-Based Sharp Bend & Corner Detection | FUOYE")
