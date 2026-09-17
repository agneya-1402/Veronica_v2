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
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
COHERE_MODEL = "command-a-plus-05-2026"
GEMINI_MODEL = "gemini-2.5-flash-lite"

VISION_FPS = 3.0
FACE_DETECT_EVERY_N_FRAMES = 2
YOLO_EVERY_N_FRAMES = 6
FACE_LOST_TIMEOUT = 2.2
FACE_MATCH_THRESHOLD = 0.593
RECOGNIZE_UNKNOWN_EVERY = 2.0
MAX_IMAGE_WIDTH = 640
RECOGNITION_MIN_FACE = 70
FACE_REACQUIRE_CENTER_JUMP = 0.24
FACE_REACQUIRE_SIZE_RATIO = 1.45
VISUAL_CONTEXT_TURNS = 6
MAX_CONTEXT_MESSAGES = 24  # 12 complete user/assistant turns
CONVERSATION_TTL = 6 * 60 * 60

cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY) if cohere and COHERE_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if genai and GEMINI_API_KEY else None

state_lock = threading.RLock()
vision_state = {
    "face": None,
    "name": "Unknown",
    "face_distance": None,
    "objects": [],
    "scene": "",
    "last_face_seen": 0.0,
    "tracking": False,
    "vision_frames": 0,
    "frame_width": 640,
    "frame_height": 480,
}

frame_counter = 0
last_face_detection_frame = -10000
last_face_recognition = 0.0
recognition_in_progress = False
yolo_in_progress = False

track_bbox: Optional[Tuple[int, int, int, int]] = None
track_points: Optional[np.ndarray] = None
track_gray: Optional[np.ndarray] = None
last_published_bbox: Optional[Tuple[int, int, int, int]] = None

FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)
if FACE_CASCADE.empty():
    raise RuntimeError("OpenCV Haar cascade could not be loaded.")

known_faces: List[Dict] = []
known_lock = threading.Lock()

# In-memory conversation store. The browser owns a stable session id in localStorage.
conversations: "OrderedDict[str, Dict]" = OrderedDict()
conversation_lock = threading.RLock()


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

    # Load the SFace model once before indexing, avoiding repeated model initialization.
    try:
        DeepFace.build_model("SFace")
    except Exception as exc:
        print(f"[Veronica] SFace preload warning: {exc}")

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
            # If an image contains multiple faces, use the largest returned face.
            rep = max(reps, key=lambda r: float(r.get("facial_area", {}).get("w", 0)) * float(r.get("facial_area", {}).get("h", 0)))
            embedding = np.asarray(rep["embedding"], dtype=np.float32)
            built.append({"name": display_name_from_file(path), "embedding": embedding})
            print(f"  + {path.name}")
        except Exception as exc:
            print(f"  ! Could not index {path.name}: {exc}")

    with known_lock:
        known_faces = built
    print(f"[Veronica] Face database ready: {len(known_faces)} identity/identities.")


def recognize_face(face_bgr: np.ndarray) -> Tuple[str, float]:
    """Recognize an already-cropped face with SFace.

    The live image has already been located by OpenCV, so DeepFace is told to
    skip a second detector pass. That avoids the common failure mode where
    OpenCV finds a face, the crop is sent back through another detector, and
    that second detector rejects the tight crop.
    """
    if DeepFace is None or face_bgr is None or face_bgr.size == 0:
        return "Unknown", 1.0
    with known_lock:
        candidates = list(known_faces)
    if not candidates:
        return "Unknown", 1.0

    try:
        h, w = face_bgr.shape[:2]
        if h < RECOGNITION_MIN_FACE or w < RECOGNITION_MIN_FACE:
            return "Unknown", 1.0

        # Give SFace a little context around the detected face while still
        # avoiding another detector pass.
        pad_x = int(w * 0.28)
        pad_y = int(h * 0.32)
        padded = cv2.copyMakeBorder(
            face_bgr, pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE,
        )
        reps = DeepFace.represent(
            img_path=padded,
            model_name="SFace",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
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
        return "Unknown", best_distance
    except Exception as exc:
        print(f"[Veronica] Recognition error: {type(exc).__name__}: {exc}")
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
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=5, minSize=(55, 55))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    # Keep the actual face box for recognition; only the tracker uses its exact box.
    return clamp_bbox((x, y, w, h), frame.shape[1], frame.shape[0])


def seed_tracker(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    global track_bbox, track_points, track_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = bbox
    mask = np.zeros_like(gray)
    mx, my = max(3, int(w * .10)), max(3, int(h * .10))
    cv2.rectangle(mask, (x + mx, y + my), (x + w - mx, y + h - my), 255, -1)
    points = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=.008, minDistance=4, blockSize=7, mask=mask)
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
    good_old = track_points[status.ravel() == 1].reshape(-1, 2)
    good_new = next_points[status.ravel() == 1].reshape(-1, 2)
    if len(good_new) < 6:
        reset_tracker()
        return None
    deltas = good_new - good_old
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))
    x, y, w, h = track_bbox
    if abs(dx) > frame.shape[1] * .12 or abs(dy) > frame.shape[0] * .12:
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
                vision_state["face_distance"] = None
            return
        x, y, w, h = box
        fw = max(1, vision_state["frame_width"])
        fh = max(1, vision_state["frame_height"])
        vision_state["face"] = {
            "x": x, "y": y, "w": w, "h": h,
            "cx": x + w / 2, "cy": y + h / 2,
            "nx": (x + w / 2) / fw, "ny": (y + h / 2) / fh,
        }
        vision_state["tracking"] = True
        vision_state["last_face_seen"] = now


def recognition_worker(crop: np.ndarray) -> None:
    global recognition_in_progress, last_face_recognition
    try:
        name, distance = recognize_face(crop)
        with state_lock:
            vision_state["name"] = name
            vision_state["face_distance"] = round(float(distance), 4) if distance < 1 else None
        print(f"[Veronica] Face recognition: {name} (cosine distance={distance:.4f})")
    finally:
        recognition_in_progress = False
        last_face_recognition = time.monotonic()


def _face_change_requires_recognition(previous, current, frame_w, frame_h) -> bool:
    if previous is None:
        return True
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    pcx, pcy = px + pw / 2, py + ph / 2
    ccx, ccy = cx + cw / 2, cy + ch / 2
    center_jump = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5 / max(1.0, (frame_w ** 2 + frame_h ** 2) ** 0.5)
    size_ratio = max(cw / max(1, pw), pw / max(1, cw), ch / max(1, ph), ph / max(1, ch))
    return center_jump > FACE_REACQUIRE_CENTER_JUMP or size_ratio > FACE_REACQUIRE_SIZE_RATIO


def update_face_tracking(frame: np.ndarray) -> None:
    global last_face_detection_frame, recognition_in_progress, last_face_recognition, last_published_bbox
    now = time.monotonic()
    frame_h, frame_w = frame.shape[:2]

    tracked = update_klt_tracker(frame)
    if tracked is not None:
        publish_face(tracked, now)

    # Haar reacquisition runs at ~1.5 FPS while KLT bridges the frames between it.
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

    changed_person_or_scale = _face_change_requires_recognition(last_published_bbox, detected, frame_w, frame_h)
    if changed_person_or_scale:
        with state_lock:
            vision_state["name"] = "Unknown"
            vision_state["face_distance"] = None
        last_face_recognition = 0.0

    # Fresh detector result is authoritative and reseeds optical flow.
    seed_tracker(frame, detected)
    publish_face(detected, now)
    last_published_bbox = detected

    with state_lock:
        current_name = vision_state["name"]
        current_distance = vision_state["face_distance"]

    should_recognize = changed_person_or_scale or current_name == "Unknown" or current_distance is None
    if should_recognize and not recognition_in_progress and (now - last_face_recognition) >= RECOGNIZE_UNKNOWN_EVERY:
        x, y, w, h = detected
        # Expand the crop; recognition itself uses detector_backend=skip.
        pad_x, pad_y = int(w * 0.22), int(h * 0.26)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(frame_w, x + w + pad_x), min(frame_h, y + h + pad_y)
        crop = frame[y0:y1, x0:x1].copy()
        if crop.shape[0] >= RECOGNITION_MIN_FACE and crop.shape[1] >= RECOGNITION_MIN_FACE:
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
            result = yolo_model.predict(frame, imgsz=416, conf=0.35, verbose=False, device=device)[0]
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
    return "I received a response, but I couldn't read its text content."


def extract_stream_text(event) -> str:
    try:
        if getattr(event, "type", None) != "content-delta":
            return ""
        delta = getattr(event, "delta", None)
        message = getattr(delta, "message", None) if delta else None
        content = getattr(message, "content", None) if message else None
        text = getattr(content, "text", None) if content else None
        return text or ""
    except Exception:
        return ""


def cleanup_conversations() -> None:
    cutoff = time.time() - CONVERSATION_TTL
    with conversation_lock:
        stale = [sid for sid, data in conversations.items() if data["updated"] < cutoff]
        for sid in stale:
            conversations.pop(sid, None)
        while len(conversations) > 100:
            conversations.popitem(last=False)


def get_session_id() -> str:
    sid = (request.headers.get("X-Veronica-Session") or "").strip()
    if not sid or len(sid) > 100:
        sid = str(uuid.uuid4())
    return sid


def get_conversation(sid: str) -> List[Dict[str, str]]:
    cleanup_conversations()
    with conversation_lock:
        item = conversations.get(sid)
        if not item:
            conversations[sid] = {"messages": [], "updated": time.time()}
            return []
        item["updated"] = time.time()
        conversations.move_to_end(sid)
        return list(item["messages"])


def append_turn(sid: str, user_text: str, assistant_text: str) -> None:
    with conversation_lock:
        item = conversations.setdefault(sid, {"messages": [], "updated": time.time()})
        messages = item["messages"]
        messages.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        # Keep whole turns only; never leave a dangling user or assistant message.
        if len(messages) > MAX_CONTEXT_MESSAGES:
            del messages[:-MAX_CONTEXT_MESSAGES]
        item["updated"] = time.time()
        conversations.move_to_end(sid)


def current_identity() -> str:
    with state_lock:
        return vision_state["name"] if vision_state["name"] != "Unknown" else ""


def build_system_prompt() -> str:
    identity = current_identity()
    identity_context = (
        f"The camera currently identifies the person in front of you as {identity}. "
        "You may address them by that name naturally."
        if identity else
        "The camera has not confidently identified the person. Do not guess their name."
    )
    return (
        "You are Veronica, a sharp, warm robotics assistant. "
        "Speak naturally, like a real conversational assistant. Be concise unless the user asks for depth. "
        "Never mention internal APIs, SDK objects, hidden reasoning, tokens, or implementation details. "
        "Never fabricate facts. If something is uncertain, say so. " + identity_context
    )


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
            "face_distance": vision_state["face_distance"],
            "objects": vision_state["objects"],
            "tracking": vision_state["tracking"],
        })


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
            response = cohere_client.chat_stream(model=COHERE_MODEL, messages=messages)
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
            # Client stopped reading. The upstream SDK may finish independently; no turn is saved.
            return
        except Exception as exc:
            print(f"[Veronica] Chat stream error: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in previous[-MAX_CONTEXT_MESSAGES:]
    )
    prompt = (
        build_system_prompt()
        + "\nUse Google Search grounding for current or web-dependent questions. "
          "Answer only from grounded information when search is needed.\n\n"
        + (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
        + f"USER: {message}"
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        reply = (response.text or "").strip()
        append_turn(sid, message, reply)
        return jsonify({"reply": reply, "provider": "gemini-google-search"})
    except Exception as exc:
        print(f"[Veronica] Web chat error: {exc}")
        return jsonify({"error": str(exc)}), 502


@app.post("/api/chat/reset")
def chat_reset():
    sid = get_session_id()
    with conversation_lock:
        conversations.pop(sid, None)
    return jsonify({"ok": True})


def is_visual_question(text: str) -> bool:
    t = text.lower().strip()
    visual_phrases = (
        "what am i holding", "what am i carrying", "what is in my hand",
        "what's in my hand", "what am i wearing", "what do you see",
        "what is happening", "what's happening", "what is going on",
        "what's going on", "describe what you see", "look at this",
        "can you see", "what is this", "what's this", "what object",
        "where am i", "who is in front of you", "what is behind me",
        "what's behind me", "is there a", "do you see a", "how many people",
        "how many", "what color is", "what colour is", "read this",
        "read the text", "can you read", "am i wearing", "is there anything"
    )
    return any(p in t for p in visual_phrases)


def visual_question_prompt(question: str, previous: List[Dict[str, str]]) -> str:
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in previous[-VISUAL_CONTEXT_TURNS * 2:]
    )
    return (
        "You are Veronica's visual perception module. Answer the user's question using ONLY the single current camera frame attached to this request. "
        "Do not use memory of earlier frames, live tracking data, object detector labels, or assumptions to invent visual facts. "
        "Conversation history may be used only to resolve a pronoun or reference such as 'it' or 'that'; every visual claim must be supported by the attached frame. "
        "If the requested thing is not visible or cannot be determined confidently from this frame, say so plainly. "
        "Do not identify a person by name from the image. Keep the answer natural and concise.\n\n"
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
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                visual_question_prompt(question, previous),
            ],
        )
        reply = (response.text or "I can't determine that from this frame.").strip()
        append_turn(sid, question, reply)
        return jsonify({"reply": reply, "provider": "gemini-vision", "frame_only": True})
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
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/jpeg"),
                (
                    "Analyze ONLY this exact camera frame. Do not use previous frames or assumptions. "
                    "Describe only clearly visible people, objects, readable text, and spatial relationships. "
                    "If something cannot be determined from this frame, say that it is not clear. "
                    "Do not invent identities, objects, actions, or locations. Keep the answer concise and useful to a person looking at the camera."
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
