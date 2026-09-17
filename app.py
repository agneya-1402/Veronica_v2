import os
import re
import time
import json
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

load_dotenv()

# Optional SDKs. Veronica remains bootable if one provider is unavailable.
try:
    import cohere
except Exception as exc:
    cohere = None
    print(f"[Veronica] Cohere import failed: {exc}")

try:
    from google import genai
    from google.genai import types
except Exception as exc:
    genai = None
    types = None
    print(f"[Veronica] Google GenAI import failed: {exc}")

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    print(f"[Veronica] Ultralytics import failed: {exc}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "known_faces"
KNOWN_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

COHERE_MODEL = os.getenv("COHERE_MODEL", "command-a-plus-05-2026")
# Stable model. Google currently documents this model as supporting Search grounding.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

VISION_FPS = 3.0
VISION_INTERVAL = 1.0 / VISION_FPS
CAMERA_CAPTURE_FPS = 10.0

MAX_IMAGE_WIDTH = 640

# YuNet settings. 2026may is the current OpenCV Zoo dynamic-input model.
YUNET_MODEL_NAME = "face_detection_yunet_2026may.onnx"
YUNET_MODEL_URL = (
    "https://huggingface.co/pollen-robotics/face_detection_yunet_2026may/"
    "resolve/main/face_detection_yunet_2026may.onnx"
)
YUNET_PATH = BASE_DIR / YUNET_MODEL_NAME
FACE_SCORE_THRESHOLD = 0.65
FACE_NMS_THRESHOLD = 0.30
FACE_TOP_K = 50

# OpenCV SFace. This avoids DeepFace/TensorFlow being in the hot webcam loop.
SFACE_MODEL_NAME = "face_recognition_sface_2021dec.onnx"
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SFACE_PATH = BASE_DIR / SFACE_MODEL_NAME

# OpenCV's documented cosine threshold for SFace is about 0.363 similarity.
# We compare distance = 1 - cosine similarity, so lower is better.
SFACE_MAX_COSINE_DISTANCE = 0.363

FACE_LOST_TIMEOUT = 1.8
RECOGNIZE_EVERY = 1.25
RECOGNITION_MIN_FACE = 45
FACE_DETECT_EVERY = 1

YOLO_EVERY_N_PROCESSED = 4
YOLO_CONF = 0.35

MAX_CONTEXT_MESSAGES = 24  # 12 complete turns
CONVERSATION_TTL = 6 * 60 * 60
VISUAL_CONTEXT_TURNS = 6


# ---------------------------------------------------------------------------
# Clients / models
# ---------------------------------------------------------------------------

cohere_client = (
    cohere.ClientV2(api_key=COHERE_API_KEY)
    if cohere and COHERE_API_KEY
    else None
)

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if genai and GEMINI_API_KEY
    else None
)

yunet = None
sface = None
yolo_model = None

state_lock = threading.RLock()
vision_state = {
    "face": None,
    "name": "Unknown",
    "face_distance": None,
    "face_score": None,
    "objects": [],
    "scene": "",
    "last_face_seen": 0.0,
    "tracking": False,
    "vision_frames": 0,
    "processed_sequence": 0,
    "frame_width": 640,
    "frame_height": 480,
    "detector": "YuNet",
}

known_faces: List[Dict] = []
known_lock = threading.RLock()

conversations: "OrderedDict[str, Dict]" = OrderedDict()
conversation_lock = threading.RLock()

# Latest-frame mailbox.
# The browser can send 10 FPS while the vision worker consumes only the newest
# frame at ~3 FPS. There is deliberately no frame queue.
latest_lock = threading.Lock()
latest_frame: Optional[np.ndarray] = None
latest_sequence = 0

worker_stop = threading.Event()
worker_thread = None

recognition_lock = threading.Lock()
recognition_in_progress = False
last_face_recognition = 0.0
last_recognized_bbox: Optional[Tuple[int, int, int, int]] = None

yolo_lock = threading.Lock()
yolo_in_progress = False
last_yolo_sequence = -1

track_bbox: Optional[Tuple[int, int, int, int]] = None
track_points: Optional[np.ndarray] = None
track_gray: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def download_if_missing(path: Path, url: str) -> bool:
    if path.exists() and path.stat().st_size > 10000:
        return True

    print(f"[Veronica] Downloading {path.name} ...")
    try:
        import urllib.request
        tmp = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size < 10000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("downloaded file is unexpectedly small")
        tmp.replace(path)
        print(f"[Veronica] Downloaded {path.name} ({path.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as exc:
        print(f"[Veronica] Could not download {path.name}: {exc}")
        return False


def display_name_from_file(path: Path) -> str:
    return re.sub(r"\s+", " ", path.stem.replace("_", " ")).strip()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    similarity = float(np.dot(a, b) / denom)
    return 1.0 - similarity


def clamp_bbox(box, width, height):
    x, y, w, h = [int(round(v)) for v in box]
    x = max(0, min(x, max(0, width - 1)))
    y = max(0, min(y, max(0, height - 1)))
    w = max(2, min(w, width - x))
    h = max(2, min(h, height - y))
    return x, y, w, h


def expand_box(box, frame_shape, px=0.18, py=0.22):
    h, w = frame_shape[:2]
    x, y, bw, bh = box
    dx = int(bw * px)
    dy = int(bh * py)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(w, x + bw + dx)
    y1 = min(h, y + bh + dy)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Face models: YuNet detector + SFace recognizer
# ---------------------------------------------------------------------------

def init_face_models():
    global yunet, sface

    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError(
            "Your OpenCV build does not provide FaceDetectorYN_create. "
            "Install a current opencv-python package in this venv."
        )

    if not download_if_missing(YUNET_PATH, YUNET_MODEL_URL):
        raise RuntimeError(
            f"YuNet model missing: {YUNET_PATH}. "
            "Download the official OpenCV Zoo YuNet model and place it beside app.py."
        )

    if not hasattr(cv2, "FaceRecognizerSF_create"):
        raise RuntimeError(
            "Your OpenCV build does not provide FaceRecognizerSF_create. "
            "Install a current opencv-python package in this venv."
        )

    if not download_if_missing(SFACE_PATH, SFACE_MODEL_URL):
        raise RuntimeError(
            f"SFace model missing: {SFACE_PATH}. "
            "Download the OpenCV Zoo SFace model and place it beside app.py."
        )

    yunet = cv2.FaceDetectorYN_create(
        str(YUNET_PATH),
        "",
        (320, 320),
        FACE_SCORE_THRESHOLD,
        FACE_NMS_THRESHOLD,
        FACE_TOP_K,
    )
    sface = cv2.FaceRecognizerSF_create(str(SFACE_PATH), "")

    print(f"[Veronica] YuNet ready: {YUNET_MODEL_NAME}")
    print(f"[Veronica] SFace ready: {SFACE_MODEL_NAME}")


def detect_faces(frame: np.ndarray) -> List[Dict]:
    """Detect all faces with YuNet and return bbox + landmarks + confidence."""
    if yunet is None:
        return []

    h, w = frame.shape[:2]
    # YuNet supports setting the actual input size at detection time.
    yunet.setInputSize((w, h))
    _, faces = yunet.detect(frame)

    if faces is None or len(faces) == 0:
        return []

    result = []
    for row in faces:
        x, y, bw, bh = row[:4]
        score = float(row[14])
        box = clamp_bbox((x, y, bw, bh), w, h)
        if box[2] < 10 or box[3] < 10:
            continue
        result.append({
            "box": box,
            "landmarks": row[4:14].astype(np.float32).reshape(5, 2),
            "score": score,
        })

    result.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    return result


def choose_face(faces: List[Dict], previous: Optional[Tuple[int, int, int, int]], frame_shape):
    if not faces:
        return None

    if previous is None:
        return faces[0]

    px, py, pw, ph = previous
    pcx, pcy = px + pw / 2, py + ph / 2

    def score(item):
        x, y, w, h = item["box"]
        cx, cy = x + w / 2, y + h / 2
        center_dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
        size_diff = abs(np.log((w * h + 1) / (pw * ph + 1)))
        return center_dist + 90.0 * size_diff - 40.0 * item["score"]

    return min(faces, key=score)


def extract_sface_embedding(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    if sface is None or face_bgr is None or face_bgr.size == 0:
        return None

    h, w = face_bgr.shape[:2]
    if min(h, w) < RECOGNITION_MIN_FACE:
        return None

    # Since this crop is already produced by YuNet, SFace can work directly
    # on the crop without running another face detector.
    try:
        feature = sface.feature(face_bgr)
        if feature is None or feature.size == 0:
            return None
        return np.asarray(feature, dtype=np.float32).reshape(-1)
    except Exception as exc:
        print(f"[Veronica] SFace feature error: {type(exc).__name__}: {exc}")
        return None


def load_known_faces():
    global known_faces

    files = sorted(
        p for p in KNOWN_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print(f"[Veronica] Indexing {len(files)} known face image(s)...")
    built = []

    for path in files:
        try:
            image = cv2.imread(str(path))
            if image is None:
                print(f"  ! Could not read {path.name}")
                continue

            faces = detect_faces(image)
            if not faces:
                print(f"  ! No face detected in {path.name}")
                continue

            face = faces[0]
            x0, y0, x1, y1 = expand_box(face["box"], image.shape, 0.10, 0.12)
            crop = image[y0:y1, x0:x1]
            embedding = extract_sface_embedding(crop)

            if embedding is None:
                print(f"  ! Could not embed {path.name}")
                continue

            built.append({
                "name": display_name_from_file(path),
                "embedding": embedding,
            })
            print(f"  + {path.name}  [{len(embedding)}D]")
        except Exception as exc:
            print(f"  ! Could not index {path.name}: {type(exc).__name__}: {exc}")

    with known_lock:
        known_faces = built

    print(f"[Veronica] Face database ready: {len(known_faces)} identity/identities.")


def recognize_crop(crop: np.ndarray) -> Tuple[str, float]:
    with known_lock:
        candidates = list(known_faces)

    if not candidates:
        return "Unknown", 1.0

    probe = extract_sface_embedding(crop)
    if probe is None:
        return "Unknown", 1.0

    best_name = "Unknown"
    best_distance = 1.0

    for item in candidates:
        d = cosine_distance(probe, item["embedding"])
        if d < best_distance:
            best_name = item["name"]
            best_distance = d

    if best_distance <= SFACE_MAX_COSINE_DISTANCE:
        return best_name, best_distance
    return "Unknown", best_distance


# ---------------------------------------------------------------------------
# Fast face tracking
# ---------------------------------------------------------------------------

def reset_tracker():
    global track_bbox, track_points, track_gray
    track_bbox = None
    track_points = None
    track_gray = None


def seed_klt(frame, bbox):
    global track_bbox, track_points, track_gray

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = bbox

    mask = np.zeros_like(gray)
    mx = max(3, int(w * 0.08))
    my = max(3, int(h * 0.08))
    x2 = max(x + mx + 1, x + w - mx)
    y2 = max(y + my + 1, y + h - my)

    cv2.rectangle(mask, (x + mx, y + my), (x2, y2), 255, -1)

    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,
        qualityLevel=0.008,
        minDistance=4,
        blockSize=7,
        mask=mask,
    )

    track_bbox = bbox
    track_points = points.reshape(-1, 1, 2) if points is not None and len(points) >= 6 else None
    track_gray = gray


def update_klt(frame) -> Optional[Tuple[int, int, int, int]]:
    global track_bbox, track_points, track_gray

    if track_bbox is None or track_points is None or track_gray is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            track_gray,
            gray,
            track_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                20,
                0.03,
            ),
        )
    except cv2.error:
        reset_tracker()
        return None

    if next_points is None or status is None:
        reset_tracker()
        return None

    good_old = track_points[status.ravel() == 1].reshape(-1, 2)
    good_new = next_points[status.ravel() == 1].reshape(-1, 2)

    # Critical: explicitly flatten both arrays to (N, 2).
    if len(good_new) < 6 or len(good_old) < 6:
        reset_tracker()
        return None

    deltas = good_new - good_old
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))

    h, w = frame.shape[:2]
    if abs(dx) > w * 0.12 or abs(dy) > h * 0.12:
        reset_tracker()
        return None

    x, y, bw, bh = track_bbox
    track_bbox = clamp_bbox((x + dx, y + dy, bw, bh), w, h)

    track_points = good_new.reshape(-1, 1, 2)
    track_gray = gray
    return track_bbox


def publish_face(box, now, score=None):
    with state_lock:
        if box is None:
            if now - vision_state["last_face_seen"] > FACE_LOST_TIMEOUT:
                vision_state["face"] = None
                vision_state["tracking"] = False
                vision_state["name"] = "Unknown"
                vision_state["face_distance"] = None
                vision_state["face_score"] = None
            return

        x, y, w, h = box
        fw = max(1, vision_state["frame_width"])
        fh = max(1, vision_state["frame_height"])

        vision_state["face"] = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "cx": x + w / 2,
            "cy": y + h / 2,
            "nx": (x + w / 2) / fw,
            "ny": (y + h / 2) / fh,
        }
        vision_state["tracking"] = True
        vision_state["last_face_seen"] = now
        if score is not None:
            vision_state["face_score"] = round(float(score), 3)


def recognition_worker(crop):
    global recognition_in_progress, last_face_recognition

    try:
        name, distance = recognize_crop(crop)
        with state_lock:
            vision_state["name"] = name
            vision_state["face_distance"] = round(float(distance), 4) if distance < 1 else None
        print(f"[Veronica] SFace recognition: {name} (distance={distance:.4f})")
    except Exception as exc:
        print(f"[Veronica] Recognition worker error: {type(exc).__name__}: {exc}")
    finally:
        recognition_in_progress = False
        last_face_recognition = time.monotonic()


def update_face_tracking(frame):
    global recognition_in_progress, last_face_recognition, last_recognized_bbox

    now = time.monotonic()
    h, w = frame.shape[:2]

    tracked = update_klt(frame)
    if tracked is not None:
        publish_face(tracked, now)

    # Run YuNet every processed frame. At ~3 FPS this is still lightweight and
    # provides reliable reacquisition rather than letting KLT drift forever.
    faces = detect_faces(frame)
    selected = choose_face(faces, tracked, frame.shape)

    if selected is None:
        if tracked is None:
            publish_face(None, now)
        return

    detected = selected["box"]
    score = selected["score"]

    # YuNet is authoritative. Reseed KLT from the fresh detection.
    seed_klt(frame, detected)
    publish_face(detected, now, score)

    # Recognition is independent of tracking and runs in a background thread.
    global_state_name = None
    with state_lock:
        global_state_name = vision_state["name"]

    should_recognize = (
        global_state_name == "Unknown"
        or last_recognized_bbox is None
        or time.monotonic() - last_face_recognition >= RECOGNIZE_EVERY
    )

    if should_recognize and not recognition_in_progress:
        x0, y0, x1, y1 = expand_box(detected, frame.shape, 0.08, 0.10)
        crop = frame[y0:y1, x0:x1].copy()

        if crop.shape[0] >= RECOGNITION_MIN_FACE and crop.shape[1] >= RECOGNITION_MIN_FACE:
            recognition_in_progress = True
            last_recognized_bbox = detected
            threading.Thread(
                target=recognition_worker,
                args=(crop,),
                daemon=True,
            ).start()


# ---------------------------------------------------------------------------
# Latest-frame mailbox and workers
# ---------------------------------------------------------------------------

def submit_latest_frame(image: np.ndarray) -> int:
    global latest_frame, latest_sequence
    with latest_lock:
        latest_sequence += 1
        latest_frame = image
        return latest_sequence


def get_latest_frame():
    with latest_lock:
        if latest_frame is None:
            return None, latest_sequence
        return latest_frame.copy(), latest_sequence


def run_yolo_async(frame, sequence):
    global yolo_in_progress, last_yolo_sequence

    if yolo_model is None:
        return
    if yolo_in_progress:
        return

    yolo_in_progress = True
    last_yolo_sequence = sequence

    def worker():
        global yolo_in_progress
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            result = yolo_model.predict(
                frame,
                imgsz=416,
                conf=YOLO_CONF,
                verbose=False,
                device=device,
            )[0]

            objects = []
            names = result.names

            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0].detach().cpu().item())
                    cls = int(box.cls[0].detach().cpu().item())
                    objects.append({
                        "label": names[cls],
                        "confidence": round(conf, 2),
                        "box": box.xyxy[0].detach().cpu().numpy().astype(int).tolist(),
                    })

            # Don't let an older YOLO job overwrite newer state.
            with latest_lock:
                newest = latest_sequence

            if sequence >= newest - 1:
                with state_lock:
                    vision_state["objects"] = objects[:30]
        except Exception as exc:
            print(f"[Veronica] YOLO error: {type(exc).__name__}: {exc}")
        finally:
            yolo_in_progress = False

    threading.Thread(target=worker, daemon=True).start()


def vision_worker():
    print(f"[Veronica] Vision worker running at {VISION_FPS:.1f} FPS; latest-frame dropping enabled.")
    processed = 0
    last_seq = -1

    while not worker_stop.is_set():
        started = time.monotonic()

        frame, sequence = get_latest_frame()
        if frame is not None and sequence != last_seq:
            last_seq = sequence

            h, w = frame.shape[:2]
            with state_lock:
                vision_state["frame_width"] = w
                vision_state["frame_height"] = h
                vision_state["processed_sequence"] = sequence
                vision_state["vision_frames"] += 1

            try:
                update_face_tracking(frame)
                processed += 1

                if processed % YOLO_EVERY_N_PROCESSED == 0:
                    run_yolo_async(frame.copy(), sequence)
            except Exception as exc:
                print(f"[Veronica] Vision worker error: {type(exc).__name__}: {exc}")

        elapsed = time.monotonic() - started
        time.sleep(max(0.005, VISION_INTERVAL - elapsed))


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def cleanup_conversations():
    cutoff = time.time() - CONVERSATION_TTL
    with conversation_lock:
        stale = [sid for sid, data in conversations.items() if data["updated"] < cutoff]
        for sid in stale:
            conversations.pop(sid, None)
        while len(conversations) > 100:
            conversations.popitem(last=False)


def get_session_id():
    sid = (request.headers.get("X-Veronica-Session") or "").strip()
    if not sid or len(sid) > 100:
        sid = str(uuid.uuid4())
    return sid


def get_conversation(sid):
    cleanup_conversations()
    with conversation_lock:
        item = conversations.get(sid)
        if not item:
            conversations[sid] = {"messages": [], "updated": time.time()}
            return []
        item["updated"] = time.time()
        conversations.move_to_end(sid)
        return list(item["messages"])


def append_turn(sid, user_text, assistant_text):
    with conversation_lock:
        item = conversations.setdefault(
            sid, {"messages": [], "updated": time.time()}
        )
        messages = item["messages"]
        messages.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        if len(messages) > MAX_CONTEXT_MESSAGES:
            del messages[:-MAX_CONTEXT_MESSAGES]
        item["updated"] = time.time()
        conversations.move_to_end(sid)


def current_identity():
    with state_lock:
        name = vision_state["name"]
        return name if name != "Unknown" else ""


def build_system_prompt():
    identity = current_identity()
    identity_context = (
        f"The camera currently identifies the person in front of you as {identity}. "
        "You may address them naturally by that name."
        if identity
        else
        "The camera has not confidently identified the person. Never guess their name."
    )

    return (
        "You are Veronica, a sharp, warm robotics assistant. "
        "Speak naturally and conversationally. Be concise unless the user asks for depth. "
        "Never mention internal APIs, SDK objects, hidden reasoning, tokens, or implementation details. "
        "Never fabricate facts. If something is uncertain, say so plainly. "
        + identity_context
    )


def is_visual_question(text):
    t = (text or "").lower().strip()

    strong = (
        "what am i holding",
        "what am i carrying",
        "what is in my hand",
        "what's in my hand",
        "what am i wearing",
        "what do i look like",
        "what do you see",
        "what is happening",
        "what's happening",
        "what is going on",
        "what's going on",
        "describe what you see",
        "look at this",
        "can you see",
        "what is this",
        "what's this",
        "what object",
        "where am i",
        "who is in front of you",
        "what is behind me",
        "what's behind me",
        "is there a ",
        "do you see a ",
        "how many people",
        "what color is",
        "what colour is",
        "read this",
        "read the text",
        "can you read",
        "am i wearing",
        "is there anything",
        "what's around me",
        "what is around me",
        "what is on my desk",
        "what's on my desk",
    )

    # These are useful natural-language variants without forcing every "what"
    # question through vision.
    if any(p in t for p in strong):
        return True

    if re.search(r"\b(am i|do i have|is there|are there)\b", t) and (
        "wear" in t or "holding" in t or "behind" in t or "visible" in t
    ):
        return True

    return False


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------

def extract_cohere_text(response):
    try:
        content = response.message.content
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if text:
                    return text.strip()
        for block in content:
            text = getattr(block, "text", None)
            if text:
                return text.strip()
    except Exception:
        pass
    return ""


def extract_stream_text(event):
    try:
        if getattr(event, "type", None) != "content-delta":
            return ""
        delta = getattr(event, "delta", None)
        message = getattr(delta, "message", None) if delta else None
        content = getattr(message, "content", None) if message else None
        return getattr(content, "text", "") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Gemini / Google Search helpers
# ---------------------------------------------------------------------------

def extract_grounding_sources(response):
    sources = []
    seen = set()

    # Current generateContent grounding metadata.
    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            if metadata is None:
                continue

            chunks = getattr(metadata, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web is None:
                    continue
                uri = getattr(web, "uri", None)
                title = getattr(web, "title", None) or uri
                if uri and uri not in seen:
                    seen.add(uri)
                    sources.append({"title": title, "url": uri})
    except Exception:
        pass

    # Interactions-style output, if a future SDK response exposes it.
    try:
        steps = getattr(response, "steps", None) or []
        for step in steps:
            content = getattr(step, "content", None) or []
            for block in content:
                annotations = getattr(block, "annotations", None) or []
                for annotation in annotations:
                    if getattr(annotation, "type", None) == "url_citation":
                        uri = getattr(annotation, "url", None)
                        title = getattr(annotation, "title", None) or uri
                        if uri and uri not in seen:
                            seen.add(uri)
                            sources.append({"title": title, "url": uri})
    except Exception:
        pass

    return sources[:8]


def gemini_search_config():
    # This is the current Google GenAI SDK generateContent form documented by
    # Google for current Gemini models.
    return types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )


def gemini_text_with_search(prompt):
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=gemini_search_config(),
    )
    return (response.text or "").strip(), extract_grounding_sources(response)


def gemini_image_with_optional_search(data, prompt, web=False):
    config = gemini_search_config() if web else None
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=data, mime_type="image/jpeg"),
            prompt,
        ],
        config=config,
    )
    return (response.text or "").strip(), extract_grounding_sources(response)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with state_lock:
        return jsonify({
            "face": vision_state["face"],
            "name": vision_state["name"],
            "face_distance": vision_state["face_distance"],
            "face_score": vision_state["face_score"],
            "objects": vision_state["objects"],
            "scene": vision_state["scene"],
            "tracking": vision_state["tracking"],
            "vision_fps": VISION_FPS,
            "camera_capture_fps": CAMERA_CAPTURE_FPS,
            "known_faces": len(known_faces),
            "cohere": bool(cohere_client),
            "gemini": bool(gemini_client),
            "detector": "YuNet",
            "recognizer": "SFace",
            "latest_sequence": latest_sequence,
            "processed_sequence": vision_state["processed_sequence"],
        })


@app.post("/api/frame")
def frame():
    """Latest-frame mailbox endpoint.

    This endpoint intentionally does NO inference. It decodes the JPEG and
    overwrites the single mailbox slot. The background worker consumes only
    the newest frame at ~3 FPS.
    """
    uploaded = request.files.get("frame")
    if not uploaded:
        return jsonify({"error": "frame is required"}), 400

    data = uploaded.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "invalid JPEG"}), 400

    h, w = image.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / w
        image = cv2.resize(
            image,
            (MAX_IMAGE_WIDTH, int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    sequence = submit_latest_frame(image)

    with state_lock:
        result = {
            "sequence": sequence,
            "face": vision_state["face"],
            "name": vision_state["name"],
            "face_distance": vision_state["face_distance"],
            "face_score": vision_state["face_score"],
            "objects": vision_state["objects"],
            "tracking": vision_state["tracking"],
            "processed_sequence": vision_state["processed_sequence"],
        }

    return jsonify(result)


@app.post("/api/chat/stream")
def chat_stream():
    if not cohere_client:
        return jsonify({"error": "COHERE_API_KEY is not configured."}), 503

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    sid = get_session_id()

    if not message:
        return jsonify({"error": "message is required"}), 400

    previous = get_conversation(sid)
    messages = [{"role": "system", "content": build_system_prompt()}]
    messages.extend(previous[-MAX_CONTEXT_MESSAGES:])
    messages.append({"role": "user", "content": message})

    def generate():
        full_text = []
        try:
            response = cohere_client.chat_stream(
                model=COHERE_MODEL,
                messages=messages,
            )

            for event in response:
                text = extract_stream_text(event)
                if text:
                    full_text.append(text)
                    yield f"data: {json.dumps({'type': 'delta', 'text': text})}\n\n"

            final = "".join(full_text).strip()
            if final:
                append_turn(sid, message, final)

            yield f"data: {json.dumps({'type': 'done', 'reply': final})}\n\n"

        except GeneratorExit:
            return
        except Exception as exc:
            print(f"[Veronica] Chat stream error: {type(exc).__name__}: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/web")
def chat_web():
    if not gemini_client:
        return jsonify({"error": "GEMINI_API_KEY is not configured."}), 503

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    sid = get_session_id()

    if not message:
        return jsonify({"error": "message is required"}), 400

    previous = get_conversation(sid)
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in previous[-MAX_CONTEXT_MESSAGES:]
    )

    prompt = (
        build_system_prompt()
        + "\n\nWEB MODE IS ON FOR THIS REQUEST. "
        "You have the Google Search grounding tool available in this request. "
        "For this WEB request, you MUST use Google Search before answering. "
        "Do not answer by saying you cannot browse, cannot run live searches, "
        "or that your knowledge only goes up to a cutoff. That would be incorrect. "
        "Search the web for current, recent, changing, niche, or externally verifiable "
        "information, then answer from the grounded results. Include the concrete "
        "current facts you found. If Search genuinely returns no useful evidence, "
        "say that the search returned insufficient results rather than claiming you "
        "cannot browse.\n\n"
        + (f"Conversation context:\n{history}\n\n" if history else "")
        + f"USER: {message}"
    )

    try:
        reply, sources = gemini_text_with_search(prompt)

        # Gemini can occasionally produce the generic 'I can't browse' refusal even
        # with the grounding tool enabled. Retry once with an even more explicit
        # grounding instruction instead of surfacing that misleading response.
        refusal_markers = (
            "i can't browse", "i cannot browse", "i can't search",
            "i cannot search", "don't have access to live",
            "do not have access to live", "knowledge cutoff",
            "training data", "can't run live searches",
            "cannot run live searches"
        )
        if reply and not sources and any(marker in reply.lower() for marker in refusal_markers):
            retry_prompt = (
                build_system_prompt()
                + "\n\nMANDATORY GOOGLE SEARCH TEST. Use the Google Search grounding tool NOW. "
                "This is not a hypothetical question about whether you can browse. "
                "Actually perform the search, use the returned web evidence, and answer "
                "the user's request. Never tell the user that you lack live-search ability. "
                "If the tool returns no evidence, state that the search returned no useful "
                "results.\n\nUSER: " + message
            )
            reply, sources = gemini_text_with_search(retry_prompt)

        if not reply:
            reply = "Google Search returned no useful grounded answer for that request."

        append_turn(sid, message, reply)

        return jsonify({
            "reply": reply,
            "provider": "gemini-google-search",
            "sources": sources,
        })
    except Exception as exc:
        print(f"[Veronica] Web chat error: {type(exc).__name__}: {exc}")
        return jsonify({"error": str(exc)}), 502


@app.post("/api/chat/reset")
def chat_reset():
    sid = get_session_id()
    with conversation_lock:
        conversations.pop(sid, None)
    return jsonify({"ok": True})


def visual_question_prompt(question, previous):
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in previous[-VISUAL_CONTEXT_TURNS * 2:]
    )

    return (
        "You are Veronica's visual perception module. "
        "Answer the user's question using ONLY the single current camera frame "
        "attached to this request for visual facts. "
        "Do not use memory of earlier frames, live tracking labels, YOLO labels, "
        "or assumptions to invent visual facts. "
        "Conversation history may be used only to resolve references such as "
        "'it', 'that', 'the other one', or similar. "
        "Every visual claim must be supported by the attached current frame. "
        "If the requested thing is not visible or cannot be determined confidently, "
        "say so plainly. Do not identify a person by name from the image. "
        "Keep the answer natural and concise.\n\n"
        + (f"Recent conversation for reference only:\n{history}\n\n" if history else "")
        + f"USER'S CURRENT QUESTION: {question}"
    )


@app.post("/api/vision/question")
def vision_question():
    if not gemini_client:
        return jsonify({"error": "GEMINI_API_KEY is not configured."}), 503

    uploaded = request.files.get("frame")
    if not uploaded:
        return jsonify({"error": "frame is required"}), 400

    question = (request.form.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    data = uploaded.read()
    if not data:
        return jsonify({"error": "empty frame"}), 400

    sid = get_session_id()
    previous = get_conversation(sid)

    try:
        web_enabled = (request.form.get("web") or "0") == "1"
        visual_prompt = visual_question_prompt(question, previous)
        if web_enabled:
            visual_prompt += (
                "\n\nWEB MODE IS ON. Use Google Search for any current or externally verifiable "
                "part of the question. You MUST use the Google Search grounding tool for "
                "the web-dependent part. Never claim you cannot browse. Keep visual claims "
                "strictly limited to what is actually visible in this exact current frame."
            )

        reply, sources = gemini_image_with_optional_search(
            data,
            visual_prompt,
            web=web_enabled,
        )

        if not reply:
            reply = "I can't determine that confidently from this current frame."

        append_turn(sid, question, reply)

        return jsonify({
            "reply": reply,
            "provider": "gemini-vision-google-search",
            "frame_only": True,
            "sources": sources,
        })
    except Exception as exc:
        print(f"[Veronica] Visual question error: {type(exc).__name__}: {exc}")
        return jsonify({"error": str(exc)}), 502


@app.post("/api/scene")
def scene():
    if not gemini_client:
        return jsonify({"error": "GEMINI_API_KEY is not configured."}), 503

    uploaded = request.files.get("frame")
    if not uploaded:
        return jsonify({"error": "frame is required"}), 400

    data = uploaded.read()
    if not data:
        return jsonify({"error": "empty frame"}), 400

    prompt = (
        "Analyze ONLY this exact current camera frame. "
        "Describe the visible scene, important people/objects, and notable activity. "
        "Do not use earlier frames, hidden context, YOLO labels, or assumptions. "
        "Do not identify people by name. If something is uncertain or not visible, "
        "say so. Keep it useful and concise."
    )

    try:
        reply, sources = gemini_image_with_optional_search(data, prompt, web=False)
        reply = reply or "I couldn't confidently analyze this frame."

        with state_lock:
            vision_state["scene"] = reply

        return jsonify({
            "scene": reply,
            "sources": sources,
            "frame_only": True,
        })
    except Exception as exc:
        print(f"[Veronica] Scene analysis error: {type(exc).__name__}: {exc}")
        return jsonify({"error": str(exc)}), 502


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup():
    global yolo_model, worker_thread

    init_face_models()
    load_known_faces()

    if YOLO is not None:
        try:
            yolo_model = YOLO("yolo26n.pt")
            print("[Veronica] YOLO26n loaded.")
        except Exception as exc:
            print(f"[Veronica] YOLO26n unavailable: {type(exc).__name__}: {exc}")
    else:
        print("[Veronica] Ultralytics unavailable; object detection disabled.")

    worker_thread = threading.Thread(target=vision_worker, daemon=True)
    worker_thread.start()

    print("")
    print("==============================================")
    print(" Veronica")
    print("==============================================")
    print(" Latest-frame mailbox : ENABLED")
    print(f" Vision processing    : {VISION_FPS:.1f} FPS")
    print(f" Camera capture       : {CAMERA_CAPTURE_FPS:.1f} FPS")
    print(" Face detector        : YuNet")
    print(" Face recognizer      : SFace")
    print(f" Known faces          : {len(known_faces)}")
    print(f" Cohere               : {'READY' if cohere_client else 'NOT CONFIGURED'}")
    print(f" Gemini                : {'READY' if gemini_client else 'NOT CONFIGURED'}")
    print(" Google Search        : ENABLED when Gemini key is configured")
    print("==============================================")
    print("")


if __name__ == "__main__":
    startup()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False, threaded=True)
