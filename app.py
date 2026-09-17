import os
import re
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

try:
    import cohere
except Exception:
    cohere = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from deepface import DeepFace
except Exception as exc:
    DeepFace = None
    print(f"[Veronica] DeepFace import failed: {exc}")

try:
    from ultralytics import YOLO
except Exception as exc:
    YOLO = None
    print(f"[Veronica] Ultralytics import failed: {exc}")

BASE_DIR = Path(__file__).resolve().parent
KNOWN_DIR = BASE_DIR / "known_faces"
KNOWN_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
COHERE_MODEL = "command-a-plus-05-2026"
GEMINI_MODEL = "gemini-2.5-flash-lite"

VISION_FPS = 3.0
FACE_DETECT_EVERY_N_FRAMES = 3       # detector ~1x/sec at 3 FPS
YOLO_EVERY_N_FRAMES = 6              # object detector ~0.5 FPS
FACE_LOST_TIMEOUT = 1.8
FACE_MATCH_THRESHOLD = 0.593
YOLO_CONF = 0.35
MAX_IMAGE_WIDTH = 640

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

state_lock = threading.RLock()
vision_state = {
    "face": None,
    "name": "Unknown",
    "objects": [],
    "scene": "",
    "last_face_seen": 0.0,
    "tracking": False,
    "vision_frames": 0,
    "last_error": "",
}

frame_counter = 0
last_face_detection_frame = -10000
last_face_recognition = 0.0
recognition_in_progress = False
yolo_in_progress = False

track_bbox: Optional[Tuple[int, int, int, int]] = None
track_points: Optional[np.ndarray] = None
track_gray: Optional[np.ndarray] = None

FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)
if FACE_CASCADE.empty():
    raise RuntimeError("OpenCV Haar cascade could not be loaded.")

known_faces: List[Dict] = []
known_lock = threading.Lock()


def display_name_from_file(path: Path) -> str:
    return re.sub(r"\s+", " ", path.stem.replace("_", " ")).strip()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def load_known_faces() -> None:
    global known_faces
    if DeepFace is None:
        print("[Veronica] DeepFace unavailable; recognition disabled.")
        return

    files = sorted(
        p for p in KNOWN_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"[Veronica] Indexing {len(files)} known face(s)...")
    built: List[Dict] = []

    for path in files:
        try:
            reps = DeepFace.represent(
                img_path=str(path),
                model_name="SFace",
                detector_backend="opencv",
                enforce_detection=True,
                align=True,
            )
            if not reps:
                print(f"  ! No face found in {path.name}")
                continue
            embedding = np.asarray(reps[0]["embedding"], dtype=np.float32)
            built.append({"name": display_name_from_file(path), "embedding": embedding})
            print(f"  + {path.name}")
        except Exception as exc:
            print(f"  ! Could not index {path.name}: {exc}")

    with known_lock:
        known_faces = built
    print(f"[Veronica] Face database ready: {len(known_faces)} identity/identities.")


def recognize_face(face_bgr: np.ndarray) -> Tuple[str, float]:
    if DeepFace is None or face_bgr is None or face_bgr.size == 0:
        return "Unknown", 1.0
    with known_lock:
        candidates = list(known_faces)
    if not candidates:
        return "Unknown", 1.0

    try:
        reps = DeepFace.represent(
            img_path=face_bgr,
            model_name="SFace",
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
        )
        if not reps:
            return "Unknown", 1.0
        probe = np.asarray(reps[0]["embedding"], dtype=np.float32)
        best_name, best_distance = "Unknown", 1.0
        for item in candidates:
            distance = cosine_distance(probe, item["embedding"])
            if distance < best_distance:
                best_name, best_distance = item["name"], distance
        if best_distance <= FACE_MATCH_THRESHOLD:
            return best_name, best_distance
    except Exception as exc:
        print(f"[Veronica] Recognition error: {exc}")
    return "Unknown", 1.0


load_known_faces()

yolo_model = None
if YOLO is not None:
    try:
        yolo_model = YOLO("yolo26n.pt")
        print("[Veronica] YOLO26n loaded.")
    except Exception as exc:
        print(f"[Veronica] YOLO26n unavailable: {exc}")
else:
    print("[Veronica] Ultralytics unavailable; object detection disabled.")


def reset_tracker() -> None:
    global track_bbox, track_points, track_gray
    track_bbox = None
    track_points = None
    track_gray = None


def clamp_bbox(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(2, min(int(w), width - x))
    h = max(2, min(int(h), height - y))
    return x, y, w, h


def detect_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    pad = int(max(w, h) * 0.12)
    return clamp_bbox((x - pad, y - pad, w + 2 * pad, h + 2 * pad), frame.shape[1], frame.shape[0])


def seed_tracker(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    global track_bbox, track_points, track_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = bbox
    mask = np.zeros_like(gray)
    mx, my = max(3, int(w * .08)), max(3, int(h * .08))
    cv2.rectangle(mask, (x + mx, y + my), (x + w - mx, y + h - my), 255, -1)
    points = cv2.goodFeaturesToTrack(
        gray, maxCorners=80, qualityLevel=.01, minDistance=5, blockSize=7, mask=mask
    )
    track_bbox = bbox
    track_points = points.reshape(-1, 1, 2) if points is not None and len(points) >= 6 else None
    track_gray = gray


def update_klt_tracker(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    global track_bbox, track_points, track_gray
    if track_bbox is None or track_points is None or track_gray is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        track_gray, gray, track_points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03),
    )
    if next_points is None or status is None:
        reset_tracker()
        return None

    good_old = track_points[status.ravel() == 1]
    good_new = next_points[status.ravel() == 1]
    if len(good_new) < 6:
        reset_tracker()
        return None

    dx = float(np.median(good_new[:, 0] - good_old[:, 0]))
    dy = float(np.median(good_new[:, 1] - good_old[:, 1]))
    x, y, w, h = track_bbox

    if abs(dx) > frame.shape[1] * .15 or abs(dy) > frame.shape[0] * .15:
        reset_tracker()
        return None

    track_bbox = clamp_bbox((x + dx, y + dy, w, h), frame.shape[1], frame.shape[0])
    track_points = good_new.reshape(-1, 1, 2)
    track_gray = gray
    return track_bbox


def publish_face(box: Optional[Tuple[int, int, int, int]], now: float) -> None:
    with state_lock:
        if box is None:
            if now - vision_state["last_face_seen"] > FACE_LOST_TIMEOUT:
                vision_state["face"] = None
                vision_state["tracking"] = False
                vision_state["name"] = "Unknown"
            return
        x, y, w, h = box
        vision_state["face"] = {
            "x": x, "y": y, "w": w, "h": h,
            "cx": x + w / 2, "cy": y + h / 2,
            "nx": (x + w / 2) / max(1, vision_state.get("frame_width", 1)),
            "ny": (y + h / 2) / max(1, vision_state.get("frame_height", 1)),
        }
        vision_state["tracking"] = True
        vision_state["last_face_seen"] = now


def recognition_worker(crop: np.ndarray) -> None:
    global recognition_in_progress, last_face_recognition
    try:
        name, distance = recognize_face(crop)
        with state_lock:
            vision_state["name"] = name
        print(f"[Veronica] Face: {name} (distance={distance:.3f})")
    finally:
        recognition_in_progress = False
        last_face_recognition = time.monotonic()


def update_face_tracking(frame: np.ndarray) -> None:
    global last_face_detection_frame, recognition_in_progress
    now = time.monotonic()
    tracked = update_klt_tracker(frame)
    if tracked is not None:
        publish_face(tracked, now)

    if (frame_counter - last_face_detection_frame) < FACE_DETECT_EVERY_N_FRAMES:
        if tracked is None:
            publish_face(None, now)
        return

    last_face_detection_frame = frame_counter
    detected = detect_face(frame)
    if detected is None:
        if tracked is None:
            publish_face(None, now)
        return

    x, y, w, h = detected
    was_tracking = tracked is not None
    seed_tracker(frame, detected)
    publish_face(detected, now)

    # Recognition only on acquisition/reacquisition, and never blocks /api/frame.
    if not was_tracking and not recognition_in_progress and (now - last_face_recognition) > 1.0:
        crop = frame[y:y+h, x:x+w].copy()
        recognition_in_progress = True
        threading.Thread(target=recognition_worker, args=(crop,), daemon=True).start()


def run_objects_async(frame: np.ndarray) -> None:
    global yolo_in_progress
    if yolo_model is None or yolo_in_progress:
        return
    yolo_in_progress = True

    def worker() -> None:
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
                    conf = float(box.conf[0].cpu().item())
                    cls = int(box.cls[0].cpu().item())
                    objects.append({
                        "label": names[cls],
                        "confidence": round(conf, 2),
                        "box": box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    })
            with state_lock:
                vision_state["objects"] = objects[:30]
        except Exception as exc:
            print(f"[Veronica] YOLO error: {exc}")
        finally:
            yolo_in_progress = False

    threading.Thread(target=worker, daemon=True).start()


def extract_cohere_text(response) -> str:
    """Command A+ can return thinking before the final text block."""
    try:
        content = response.message.content
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if text:
                    return text.strip()
        # Defensive fallback for SDK object variants.
        for block in content:
            text = getattr(block, "text", None)
            if text:
                return text.strip()
    except Exception:
        pass
    return "I received a response, but I couldn't read its text content."


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with state_lock:
        return jsonify({
            "face": vision_state["face"],
            "name": vision_state["name"],
            "objects": vision_state["objects"],
            "scene": vision_state["scene"],
            "tracking": vision_state["tracking"],
            "vision_fps": VISION_FPS,
            "known_faces": len(known_faces),
            "cohere": bool(cohere_client),
            "gemini": bool(gemini_client),
        })


@app.post("/api/frame")
def frame():
    global frame_counter
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
        image = cv2.resize(image, (MAX_IMAGE_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]

    with state_lock:
        vision_state["frame_width"] = w
        vision_state["frame_height"] = h

    frame_counter += 1
    update_face_tracking(image)

    if frame_counter % YOLO_EVERY_N_FRAMES == 0:
        run_objects_async(image.copy())

    with state_lock:
        vision_state["vision_frames"] += 1
        return jsonify({
            "face": vision_state["face"],
            "name": vision_state["name"],
            "objects": vision_state["objects"],
            "tracking": vision_state["tracking"],
        })


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    history = payload.get("history") or []
    if not message:
        return jsonify({"error": "message is required"}), 400

    web_mode = bool(payload.get("web", False))
    try:
        if web_mode and gemini_client:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=message,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    system_instruction=(
                        "You are Veronica, a concise desktop robot assistant. "
                        "Use Google Search grounding for current or web-dependent questions. "
                        "Never invent facts. Give a direct answer and mention uncertainty when evidence is insufficient."
                    ),
                ),
            )
            return jsonify({"reply": response.text.strip(), "provider": "gemini-google-search"})

        if not cohere_client:
            return jsonify({"error": "COHERE_API_KEY is not configured."}), 503

        messages = [{
            "role": "system",
            "content": (
                "You are Veronica, a sharp, warm robotics assistant. "
                "Speak naturally and concisely. Do not mention internal APIs, SDK objects, tokens, or response metadata. "
                "Never fabricate facts."
            ),
        }]
        for item in history[-12:]:
            role = item.get("role")
            content = item.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        response = cohere_client.chat(model=COHERE_MODEL, messages=messages)
        return jsonify({"reply": extract_cohere_text(response), "provider": "cohere"})
    except Exception as exc:
        print(f"[Veronica] Chat error: {exc}")
        return jsonify({"error": str(exc)}), 502


@app.post("/api/scene")
def scene():
    if not gemini_client:
        return jsonify({"error": "GEMINI_API_KEY is not configured."}), 503
    uploaded = request.files.get("frame")
    if not uploaded:
        return jsonify({"error": "frame is required"}), 400

    data = uploaded.read()
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                (
                    "Describe the visible environment for a robot assistant. "
                    "Identify only clearly visible objects, people, text, and spatial relationships. "
                    "Do not guess or invent details. Keep it concise."
                ),
            ],
        )
        text = (response.text or "No clear scene description was returned.").strip()
        with state_lock:
            vision_state["scene"] = text
        return jsonify({"scene": text})
    except Exception as exc:
        print(f"[Veronica] Scene error: {exc}")
        return jsonify({"error": str(exc)}), 502


if __name__ == "__main__":
    print("\nVeronica running at http://127.0.0.1:8000")
    print("Add API keys to .env and identity images to known_faces/.")
    print(f"Known faces: {len(known_faces)}")
    print(f"Cohere: {'ON' if cohere_client else 'OFF'} | Gemini: {'ON' if gemini_client else 'OFF'}")
    app.run(host="127.0.0.1", port=8000, threaded=True, debug=False, use_reloader=False)
