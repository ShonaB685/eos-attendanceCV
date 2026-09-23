"""Basic web front-end for the recognition pipeline: a browser page that
captures photos via webcam and calls into the same recognize_faces() /
mark_attendance() functions the CLI (mark_attendance.py) uses, so there is
exactly one place the matching/liveness/attendance logic lives."""

import base64
import os
import threading
from collections import defaultdict
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request

from add_student import CAPTURE_INTERVAL_MS, POSES, SAMPLES_PER_POSE, SAMPLES_TO_CAPTURE, sanitize_id
from auth import get_api_key
from common import (
    CAPTURES_DIR,
    DATASET_DIR,
    cleanup_old_captures,
    delete_student,
    detect_faces,
    embed_face,
    get_antispoof_sessions,
    get_embedder,
    get_face_aligner,
    get_face_detector,
    load_gallery,
    release_memory,
    rss_mb,
    save_student,
    warmup_antispoof,
    warmup_detector,
    warmup_embedder,
)
from mark_attendance import find_existing_match, load_active_roster, mark_attendance, recognize_faces, resolve_banking
from train_model import retrain_student

app = Flask(__name__)

API_KEY = get_api_key()
if os.environ.get("ATTENDANCE_API_KEY"):
    # Never echo a real deployment's secret into (hosted) logs.
    print("API key loaded from ATTENDANCE_API_KEY.")
else:
    print(f"API key for non-browser clients (mobile app, etc.): {API_KEY}")
    print("(The web pages already know this key and send it automatically.)")


def _log_memory(stage):
    rss = rss_mb()
    if rss is not None:
        print(f"[memory] {stage}: rss={rss:.0f} MB", flush=True)


# --- Lazy, thread-safe model loading --------------------------------------
# Nothing heavy loads at import time any more: loading the whole pipeline
# up front (ArcFace alone is ~+186-272 MB) plus warming it made Gunicorn's
# worker exceed Render Free's 512 MB before it could serve a single request.
# Each component now loads on first use by the endpoint that needs it, is
# warmed right after (one component at a time, so warm-up never stacks on
# top of another model's load spike), and is then shared by every later
# request - exactly one detector / aligner / embedder / anti-spoof pair per
# process, never reloaded.
_pipeline = {}
_pipeline_lock = threading.Lock()
_WARMUP = os.environ.get("CV_WARMUP", "1") != "0"


def _load(key, loader, warmup=None):
    component = _pipeline.get(key)
    if component is not None:
        return component
    with _pipeline_lock:
        component = _pipeline.get(key)
        if component is None:
            component = loader()
            if warmup is not None and _WARMUP:
                warmup(component)
            _pipeline[key] = component
            release_memory()
            _log_memory(f"after loading {key}")
    return component


def get_detector():
    return _load("detector", get_face_detector, warmup_detector)


def get_aligner():
    return _load("aligner", get_face_aligner)


def get_embedder_session():
    return _load("embedder", get_embedder, warmup_embedder)


def get_antispoof():
    return _load("antispoof_sessions", get_antispoof_sessions, warmup_antispoof)


def get_gallery():
    """The in-memory gallery is the live copy (enrollment/banking update it
    in place); it's read from disk once, on first use."""
    return _load("gallery", load_gallery)


_log_memory("startup, before any model")
_removed = cleanup_old_captures()
if _removed:
    print(f"Cleaned up {_removed} capture photo(s) older than the retention window.")


def require_api_key(view):
    """Gates one route behind the shared API key -- checked via header so
    it works the same for the browser frontend (sends it automatically,
    see index.html/enroll.html) and any other client (mobile app, curl,
    etc.) that includes X-API-Key itself."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if request.headers.get("X-API-Key") != API_KEY:
            abort(401, description="Missing or invalid X-API-Key header.")
        return view(*args, **kwargs)
    return wrapped


def decode_image(data_url):
    """Turns a 'data:image/jpeg;base64,...' string from a <canvas> into a
    BGR frame, or None if it isn't a decodable image (missing field, empty
    string, truncated/corrupt payload -- any of those should be a graceful
    "no image" rather than a 500)."""
    if not data_url:
        return None
    b64 = data_url.split(",", 1)[-1]
    try:
        raw = base64.b64decode(b64)
    except (ValueError, TypeError):
        return None
    if not raw:
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def encode_image(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.route("/")
def index():
    roster = sorted(load_active_roster().items(), key=lambda kv: kv[1])
    return render_template("index.html", students=[name for _, name in roster], api_key=API_KEY)


@app.route("/add-student")
def add_student_page():
    roster = sorted(load_active_roster().items(), key=lambda kv: kv[1])
    return render_template(
        "enroll.html",
        poses=POSES,
        samples_per_pose=SAMPLES_PER_POSE,
        samples_to_capture=SAMPLES_TO_CAPTURE,
        capture_interval_ms=CAPTURE_INTERVAL_MS,
        api_key=API_KEY,
        students=[{"id": sid, "name": name} for sid, name in roster],
    )


@app.route("/api/enroll", methods=["POST"])
@require_api_key
def api_enroll():
    """Browser equivalent of add_student.py's main(): every captured frame
    must show exactly one face (same rule the CLI enforces) to be kept.

    Two modes, distinguished by whether the request names an existing
    student_id (re-enrollment -- "add more samples for someone already
    enrolled", picked from a list in the UI, not typed) or not (brand new
    enrollment, typed name -> sanitize_id()). Re-enrollment skips the
    duplicate-face check entirely, since identity was already confirmed by
    selecting who this is rather than inferred from a name string -- a
    fresh enrollment still runs find_existing_match() so someone already
    enrolled can't accidentally register a second time under a typo'd or
    different name. Only this one student's images get (re-)embedded, not
    the whole dataset -- a full retrain measured ~154s at 275 images,
    unusable per-enrollment."""
    payload = request.get_json(silent=True) or {}
    data_urls = payload.get("images") or []
    if not data_urls:
        return jsonify(error="No photos captured."), 400

    re_enroll_id = (payload.get("student_id") or "").strip()
    if re_enroll_id:
        roster = load_active_roster()
        if re_enroll_id not in roster:
            return jsonify(error=f"No existing student with id \"{re_enroll_id}\"."), 404
        student_id, name = re_enroll_id, roster[re_enroll_id]
    else:
        name = (payload.get("name") or "").strip()
        if not name:
            return jsonify(error="Name is required."), 400
        student_id = sanitize_id(name)

    student_dir = os.path.join(DATASET_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)
    existing = len([f for f in os.listdir(student_dir) if f.endswith(".jpg")])

    detector = get_detector()
    aligner = get_aligner()
    captured = 0
    skipped = 0
    aligned_faces = []
    for data_url in data_urls:
        frame = decode_image(data_url)
        if frame is None:
            skipped += 1
            continue
        faces = detect_faces(detector, frame)
        if len(faces) != 1:
            skipped += 1
            del frame
            continue
        aligned = aligner.alignCrop(frame, faces[0])
        # Only the 112x112 crop is kept; the full-resolution frame goes now.
        del frame
        aligned_faces.append(aligned)
        existing += 1
        captured += 1
        cv2.imwrite(os.path.join(student_dir, f"{existing}.jpg"), aligned)

    if captured == 0:
        return jsonify(error="No usable face samples captured (need exactly one clear face per shot)."), 400

    embedder = get_embedder_session()
    gallery = get_gallery()
    if not re_enroll_id:
        sample_embeddings = [embed_face(embedder, face) for face in aligned_faces[:10]]
        match_id, match_sim = find_existing_match(sample_embeddings, gallery)
        if match_id is not None and match_id != student_id:
            # New/never-registered student_dir -- safe to discard entirely.
            for f in os.listdir(student_dir):
                os.remove(os.path.join(student_dir, f))
            os.rmdir(student_dir)
            return jsonify(
                error=f"This face already appears to be enrolled as \"{match_id}\" (similarity={match_sim:.2f}). "
                      "Not registering as a new/duplicate student."
            ), 409

    save_student(student_id, name)
    gallery[student_id] = retrain_student(student_id, embedder)
    del aligned_faces
    release_memory()

    return jsonify(student_id=student_id, name=name, captured=captured, skipped=skipped)


@app.route("/api/delete-student", methods=["POST"])
@require_api_key
def api_delete_student():
    payload = request.get_json(silent=True) or {}
    student_id = (payload.get("student_id") or "").strip()
    if not student_id:
        return jsonify(error="student_id is required."), 400

    deleted = delete_student(student_id)
    if not deleted:
        return jsonify(error=f"No student with id \"{student_id}\"."), 404

    # delete_student() already removed them from gallery.pickle on disk; only
    # the in-memory copy needs syncing, and only if it was ever loaded (a
    # not-yet-loaded gallery will simply be read fresh from disk later).
    if "gallery" in _pipeline:
        _pipeline["gallery"].pop(student_id, None)
    return jsonify(student_id=student_id, deleted=True)


@app.route("/api/detect", methods=["POST"])
@require_api_key
def api_detect():
    """Detection only (no liveness/embedding/matching) -- cheap enough to
    poll continuously while the live video is on screen, so the box shows
    up before the user ever clicks anything."""
    payload = request.get_json(silent=True) or {}
    frame = decode_image(payload.get("image") or "")
    if frame is None:
        return jsonify(faces=[])

    faces = detect_faces(get_detector(), frame)
    del frame
    boxes = [
        {"x": int(f[0]), "y": int(f[1]), "w": int(f[2]), "h": int(f[3])}
        for f in faces
    ]
    return jsonify(faces=boxes)


@app.route("/api/mark", methods=["POST"])
@require_api_key
def api_mark():
    """Mirrors mark_attendance.py's main(): recognize across every captured
    photo in this batch, keep each student's best similarity, mark
    attendance once. Banking a new gallery sample requires the same
    student to have matched as a bank-candidate in multiple distinct
    photos in this batch (see resolve_banking/MIN_PHOTOS_TO_BANK) -- a
    single high-confidence frame is not enough to permanently write into
    the gallery."""
    payload = request.get_json(silent=True) or {}
    data_urls = payload.get("images") or []
    if not data_urls:
        return jsonify(error="No images captured."), 400

    if not load_active_roster():
        return jsonify(error="No active students in dataset/. Add students first."), 400

    # Loaded one at a time (each followed by its own warm-up) on the first
    # /api/mark, then reused - see _load().
    detector = get_detector()
    aligner = get_aligner()
    embedder = get_embedder_session()
    antispoof_sessions = get_antispoof()
    gallery = get_gallery()

    overall_recognized = {}
    all_candidates = defaultdict(list)
    total_spoofed = 0
    photos = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # One photo at a time: only the current full-resolution frame and its
    # annotated copy are ever alive together; each is dropped before the
    # next photo is decoded. (The encoded `photos` stay - they're part of
    # the response the browser page shows.)
    for idx, data_url in enumerate(data_urls, start=1):
        frame = decode_image(data_url)
        if frame is None:
            continue

        recognized, annotated, bank_candidates, spoofed = recognize_faces(
            frame,
            aligner,
            embedder,
            gallery,
            detector,
            antispoof_sessions,
        )
        del frame
        for student_id, similarity in recognized.items():
            if student_id not in overall_recognized or similarity > overall_recognized[student_id]:
                overall_recognized[student_id] = similarity
        for student_id, aligned_face, embedding, similarity in bank_candidates:
            all_candidates[student_id].append((aligned_face, embedding, similarity, idx))
        total_spoofed += spoofed
        photos.append(encode_image(annotated))

        # Same trail the CLI leaves in CAPTURES_DIR, so a "why was I marked
        # absent" report can actually be diagnosed after the fact.
        snapshot_path = os.path.join(CAPTURES_DIR, f"web_{timestamp}_{idx}.jpg")
        cv2.imwrite(snapshot_path, annotated)
        del annotated

    rows = mark_attendance(overall_recognized)
    total_banked = resolve_banking(gallery, all_candidates, len(data_urls), embedder)
    del all_candidates
    release_memory()

    results = [
        {"student_id": sid, "name": name, "status": status}
        for _, _, sid, name, status in rows
    ]
    return jsonify(results=results, banked=total_banked, spoofed=total_spoofed, photos=photos)


if __name__ == "__main__":
    # 127.0.0.1-only by default: this endpoint marks attendance with no
    # authentication, so keep it off the LAN until that's addressed.
    app.run(host="127.0.0.1", port=5000, debug=False)
