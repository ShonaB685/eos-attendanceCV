import csv
import os
import pickle
import shutil
import time

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
STUDENTS_CSV = os.path.join(BASE_DIR, "students.csv")
ATTENDANCE_CSV = os.path.join(BASE_DIR, "attendance.csv")
GALLERY_FILE = os.path.join(MODELS_DIR, "gallery.pickle")

# CAPTURES_DIR holds the annotated snapshot from every attendance-marking
# photo -- useful for diagnosing "why was I marked absent" right after the
# fact, but there's no reason to keep those photos of people's faces
# indefinitely. The attendance *record* (who/when/present-or-absent) lives
# forever in the DB; only the photographic evidence expires. Recognition
# accuracy and gallery data (dataset/) are untouched by this -- this is
# just the daily audit-trail images.
CAPTURE_RETENTION_DAYS = 3

YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
# SFace is kept only for its alignCrop() -- a reliable 5-point similarity-transform
# aligner. Its own embeddings/matching are unused; ArcFace below does recognition.
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
ARCFACE_MODEL = os.path.join(MODELS_DIR, "arcface_r50.onnx")

# Anti-spoofing (liveness): MiniFASNetV2 + MiniFASNetV1SE, the two-model
# ensemble from minivision-ai's Silent-Face-Anti-Spoofing (CVPR2020 challenge
# winner), ONNX-exported by github.com/yakhyo/face-anti-spoofing. V2 (crop
# scale 2.7, tighter around the face) targets print/photo attacks; V1SE (scale
# 4.0, wider context) targets screen-replay attacks -- catching a phone/tablet
# held up to the camera is exactly the V1SE half of this ensemble.
ANTISPOOF_MODELS = [
    (os.path.join(MODELS_DIR, "MiniFASNetV2.onnx"), 2.7),
    (os.path.join(MODELS_DIR, "MiniFASNetV1SE.onnx"), 4.0),
]
ANTISPOOF_INPUT_SIZE = 80

ALIGNED_FACE_SIZE = (112, 112)  # fixed input size both alignCrop and ArcFace expect

# ArcFace cosine similarity: higher = more alike. These are starting points,
# not measured values -- calibrate against real enrollment/test photos once
# there's enough data (see the accuracy bottleneck discussion: threshold
# should come from measured FAR/FRR, not a paper default).
COSINE_THRESHOLD = 0.40

# Beyond clearing the threshold, the best match must also beat the next-best
# *different* student by this margin. This is what stops two genuinely
# similar-looking people from being confused: a narrow win is treated as
# ambiguous ("Unknown") instead of confidently assigned.
MARGIN = 0.07

# Only bank a captured face as a new gallery sample when the match was
# well clear of both the threshold and the margin, so a borderline/wrong
# match can never reinforce itself into the gallery.
AUTO_ENROLL_SIMILARITY = 0.60

# A high-confidence match in a *single* photo isn't enough to permanently
# write into the gallery -- multiple photos get taken per session (moving
# around a room), and a spoof only needs to slip past liveness on one of
# those frames to poison the gallery forever. Banking requires the same
# student to win as the best match in this many *distinct* photos when
# more than one was captured (capped at however many were actually taken,
# so a single-photo session still allows banking on one confident match).
MIN_PHOTOS_TO_BANK = 2

# Caps how many auto-banked (live-match) samples a single student can
# accumulate -- unbounded growth here slows every future full retrain
# (train_model.py re-embeds every image from scratch) and bloats dataset/
# indefinitely. Original enrollment photos are never touched by this, only
# auto-banked ones once they exceed the cap (oldest evicted first).
MAX_AUTO_SAMPLES_PER_STUDENT = 60

# Pruning back down to the cap re-embeds that student's whole remaining
# image set (measured ~10s at ~85 images) -- letting the count drift up to
# MAX_AUTO_SAMPLES_PER_STUDENT + PRUNE_BATCH before pruning a whole batch
# at once means that cost is paid every ~10 events instead of every single
# one once a student first hits the cap.
PRUNE_BATCH = 10

# Minimum averaged "real" probability from the anti-spoofing ensemble for a
# face to be treated as live. Like COSINE_THRESHOLD, this needs calibration
# against real conditions rather than trusting a default -- first real data
# point (2026-08-03): a genuine live face scored 0.535 under this webcam's
# lighting (rejected at the original 0.60), while an actual phone-replay
# spoof scored 0.000 across every capture in the same session. That's a
# wide gap between real and fake, so the threshold has headroom to drop
# without risking the spoof getting through. Revisit as more real-world
# scores come in -- one borderline real sample isn't the last word.
LIVENESS_THRESHOLD = 0.40

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CAPTURES_DIR, exist_ok=True)


def cleanup_old_captures(retention_days=CAPTURE_RETENTION_DAYS):
    """Deletes snapshot photos in CAPTURES_DIR older than retention_days.
    Call this at process startup (see app.py / mark_attendance.main) --
    there's no scheduler here, just a sweep whenever the app starts, which
    is enough for a low-traffic daily-attendance workload."""
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for fname in os.listdir(CAPTURES_DIR):
        path = os.path.join(CAPTURES_DIR, fname)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            os.remove(path)
            removed += 1
    return removed


def get_face_detector(input_size=(320, 320)):
    return cv2.FaceDetectorYN.create(YUNET_MODEL, "", input_size, score_threshold=0.7)


def get_face_aligner():
    """cv2.FaceRecognizerSF used purely as a landmark-based aligner (alignCrop)."""
    return cv2.FaceRecognizerSF.create(SFACE_MODEL, "")


# Every onnxruntime session below defaults to grabbing an intra-op thread
# per CPU core. That's fine in isolation, but this pipeline keeps 3 such
# sessions (ArcFace + 2x MiniFASNet) resident at once, plus OpenCV's own
# threading for YuNet/SFace -- left uncapped they contend for the same
# cores and ArcFace inference alone measured 300-500ms instead of its
# ~110ms steady-state. 4 threads/session was the measured sweet spot on a
# 12-core machine; more made it slower, not faster.
_ORT_SESSION_OPTIONS = ort.SessionOptions()
_ORT_SESSION_OPTIONS.intra_op_num_threads = 4
_ORT_SESSION_OPTIONS.inter_op_num_threads = 1


def get_embedder():
    return ort.InferenceSession(ARCFACE_MODEL, sess_options=_ORT_SESSION_OPTIONS, providers=["CPUExecutionProvider"])


def get_antispoof_sessions():
    """Loads the anti-spoofing ensemble: a list of (onnxruntime session,
    crop scale) pairs, one per model in ANTISPOOF_MODELS."""
    return [
        (ort.InferenceSession(path, sess_options=_ORT_SESSION_OPTIONS, providers=["CPUExecutionProvider"]), scale)
        for path, scale in ANTISPOOF_MODELS
    ]


def warmup_pipeline(detector, aligner, embedder, antispoof_sessions):
    """Runs one throwaway inference through every model. ONNX Runtime pays a
    real one-time cost on a session's *first* call -- lazy memory-arena
    allocation, kernel selection -- separate from steady-state latency; for
    ArcFace specifically that's the difference between ~310ms and ~90ms.
    Call this once at process startup so that cost lands during boot instead
    of on whoever triggers the first real recognition."""
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detect_faces(detector, dummy_frame)

    dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)
    embed_face(embedder, dummy_face)

    check_liveness(antispoof_sessions, dummy_frame, (80, 60, 160, 160))


def detect_faces(detector, frame):
    """Returns an array of detections (x, y, w, h, 5x landmark pairs, score)
    or an empty list. Caller must resize the detector's input size to the
    frame first if the frame size can change between calls."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    return faces if faces is not None else []


# BLUR_VARIANCE_THRESHOLD calibrated 2026-08-06 against real webcam frames
# after the original guess (50) started rejecting clearly-usable photos.
# Three frames from one session, same camera, same person, seconds apart,
# scored 31.4 / 201.5 / 338.9 -- ~10x natural swing from ordinary frame-to-
# frame focus hunting, nothing to do with actual usability. Meanwhile
# synthetic motion blur (visibly, genuinely too blurry) scored 6.1 and
# below. There's a real gap between "normal camera noise" and "actually
# blurry" -- just not anywhere near 50. Still not exhaustively tested
# across camera qualities (phones will vary more than one laptop webcam),
# so lean conservative: a missed genuinely-blurry frame just falls through
# to the normal match-threshold failing anyway, while a false "too blurry"
# rejection blocks attendance outright.
BLUR_VARIANCE_THRESHOLD = 10.0   # Laplacian variance on the face crop
BRIGHTNESS_MIN = 30               # mean pixel intensity, 0-255
BRIGHTNESS_MAX = 225


def assess_face_quality(frame, bbox_xywh):
    """Cheap pre-checks on a detected face crop, run before the expensive
    liveness/embedding pipeline. Returns (ok, reason) -- reason is None
    when ok, otherwise a short label ('too_dark'/'too_bright'/'too_blurry')
    so a bad photo gets a clear, specific diagnosis (and a chance to
    retake) instead of just silently failing to match somewhere
    downstream with no explanation."""
    x, y, w, h = bbox_xywh
    src_h, src_w = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(src_w, x + w), min(src_h, y + h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "empty_crop"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    if brightness < BRIGHTNESS_MIN:
        return False, "too_dark"
    if brightness > BRIGHTNESS_MAX:
        return False, "too_bright"

    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_variance < BLUR_VARIANCE_THRESHOLD:
        return False, "too_blurry"

    return True, None


def _crop_for_antispoof(frame, bbox_xywh, scale, out_size):
    """Re-crops the face region around bbox_xywh at the given context scale
    (wider than the tight aligned crop ArcFace uses) and resizes to
    out_size x out_size. Mirrors Silent-Face-Anti-Spoofing's own crop logic
    exactly, since the model was trained on crops built this way."""
    src_h, src_w = frame.shape[:2]
    x, y, box_w, box_h = bbox_xywh

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    center_x, center_y = x + box_w / 2, y + box_h / 2

    x1 = max(0, int(center_x - new_w / 2))
    y1 = max(0, int(center_y - new_h / 2))
    x2 = min(src_w - 1, int(center_x + new_w / 2))
    y2 = min(src_h - 1, int(center_y + new_h / 2))

    cropped = frame[y1:y2 + 1, x1:x2 + 1]
    return cv2.resize(cropped, (out_size, out_size))


def _softmax(logits):
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def check_liveness(antispoof_sessions, frame, bbox_xywh):
    """Runs the anti-spoofing ensemble on one detected face. Returns
    (is_real, real_score): real_score is the models' averaged softmax
    probability for the "real" class (index 1); is_real additionally
    requires that class to have won (argmax) -- matching the upstream
    project's own decision rule -- on top of clearing LIVENESS_THRESHOLD.
    Catches printed photos and, notably, phone/tablet screen replay (someone
    holding up another student's photo on a device), which is what the
    wider-context V1SE model in the ensemble is specifically trained for."""
    probs_sum = None
    for session, scale in antispoof_sessions:
        face = _crop_for_antispoof(frame, bbox_xywh, scale, ANTISPOOF_INPUT_SIZE)
        blob = np.expand_dims(face.astype(np.float32).transpose(2, 0, 1), axis=0)
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: blob})[0]
        probs = _softmax(logits)
        probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_avg = probs_sum / len(antispoof_sessions)
    label = int(np.argmax(probs_avg))
    real_score = float(probs_avg[0, 1])
    is_real = label == 1 and real_score >= LIVENESS_THRESHOLD
    return is_real, real_score


def embed_face(embedder, aligned_bgr_face):
    """Runs an aligned 112x112 BGR face crop through ArcFace and returns its
    512-d embedding as a 1D numpy array. Preprocessing matches insightface's
    standard ArcFace pipeline: BGR->RGB, scale to [-1, 1], NCHW."""
    blob = cv2.dnn.blobFromImage(
        aligned_bgr_face, scalefactor=1.0 / 127.5, size=ALIGNED_FACE_SIZE,
        mean=(127.5, 127.5, 127.5), swapRB=True,
    )
    input_name = embedder.get_inputs()[0].name
    output = embedder.run(None, {input_name: blob})[0]
    return output.flatten()


def cosine_similarity(embedding_a, embedding_b):
    denom = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
    if denom == 0:
        return 0.0
    return float(np.dot(embedding_a, embedding_b) / denom)


# --- Plain-file storage: students.csv / attendance.csv / models/gallery.pickle.
# Deliberately no database here -- the deployment team owns the real data
# store (Postgres, per plan); this stays simple/local so there's nothing
# database-specific to rip out later, just a storage layer to point
# elsewhere. -----------------------------------------------------------

def load_students():
    students = {}
    if os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                students[row["id"]] = row["name"]
    return students


def save_student(student_id, name):
    students = load_students()
    if student_id in students:
        return
    write_header = not os.path.exists(STUDENTS_CSV)
    with open(STUDENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id", "name"])
        writer.writerow([student_id, name])


def load_gallery():
    if not os.path.exists(GALLERY_FILE):
        return {}
    with open(GALLERY_FILE, "rb") as f:
        return pickle.load(f)


def save_gallery(gallery):
    with open(GALLERY_FILE, "wb") as f:
        pickle.dump(gallery, f)


def delete_student(student_id):
    """Removes a student entirely: roster entry, dataset images, and
    gallery embeddings. Irreversible -- there's no undo once this runs."""
    students = load_students()
    if student_id not in students:
        return False

    del students[student_id]
    with open(STUDENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name"])
        for sid, name in students.items():
            writer.writerow([sid, name])

    student_dir = os.path.join(DATASET_DIR, student_id)
    if os.path.isdir(student_dir):
        shutil.rmtree(student_dir)

    gallery = load_gallery()
    if student_id in gallery:
        del gallery[student_id]
        save_gallery(gallery)

    return True


