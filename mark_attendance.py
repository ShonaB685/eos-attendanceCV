import csv
import os
from collections import defaultdict
from datetime import datetime

import cv2

from common import (
    ATTENDANCE_CSV,
    AUTO_ENROLL_SIMILARITY,
    CAPTURES_DIR,
    COSINE_THRESHOLD,
    DATASET_DIR,
    LIVENESS_THRESHOLD,
    MARGIN,
    MAX_AUTO_SAMPLES_PER_STUDENT,
    MIN_PHOTOS_TO_BANK,
    PRUNE_BATCH,
    assess_face_quality,
    check_liveness,
    cleanup_old_captures,
    cosine_similarity,
    detect_faces,
    embed_face,
    get_antispoof_sessions,
    get_embedder,
    get_face_aligner,
    get_face_detector,
    load_gallery,
    load_students,
    save_gallery,
    warmup_pipeline,
)
from train_model import retrain_student


def load_active_roster():
    """Students whose dataset folder still exists, so a deleted/renamed
    folder can't leave a stale ghost entry in the attendance roster."""
    students = load_students()
    return {
        student_id: name
        for student_id, name in students.items()
        if os.path.isdir(os.path.join(DATASET_DIR, student_id))
    }


def capture_photos():
    """Capture as many photos as needed (e.g. moving around the room so
    faces stay large enough to recognize, instead of one wide shot where
    distant faces are too small to detect). ENTER captures another photo,
    F finishes and moves on to processing, ESC discards everything."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam.")
        raise SystemExit(1)

    print("Take as many photos as you need to cover everyone. ENTER: capture, F: finish & process, ESC: cancel.")
    frames = []
    while True:
        ok, live = cam.read()
        if not ok:
            break
        display = live.copy()
        cv2.putText(display, f"Captured: {len(frames)}  ENTER=capture  F=finish  ESC=cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Mark Attendance - multi-photo capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            frames.append(live.copy())
            print(f"  Captured photo {len(frames)}.")
        elif key in (ord("f"), ord("F")):
            break
        elif key == 27:  # Esc
            frames = []
            break

    cam.release()
    cv2.destroyAllWindows()
    return frames


def bank_auto_sample(gallery, student_id, aligned_face, embedding, embedder):
    """Saves a high-confidence aligned face crop to disk (provenance) and
    appends its already-computed embedding to the in-memory gallery.

    Pruning back down to MAX_AUTO_SAMPLES_PER_STUDENT calls retrain_student()
    -- re-embedding that one student's entire remaining image set, not the
    whole dataset, but still real cost (measured ~10s for ~85 images).
    Triggering that on every single over-cap event would mean every
    qualifying match costs ~10s forever once a student first hits the cap
    (measured this happening -- 400ms baseline jumped to ~10s per request).
    PRUNE_BATCH lets the count drift up to MAX + PRUNE_BATCH before pruning
    a whole batch back down at once, so the expensive re-embed happens
    every ~10 events instead of every single one. Only called once
    cross-photo agreement has already been confirmed by the caller -- see
    MIN_PHOTOS_TO_BANK."""
    student_dir = os.path.join(DATASET_DIR, student_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(os.path.join(student_dir, f"auto_{timestamp}.jpg"), aligned_face)
    gallery.setdefault(student_id, []).append(embedding)

    auto_files = sorted(
        f for f in os.listdir(student_dir) if f.startswith("auto_") and f.endswith(".jpg")
    )
    excess = len(auto_files) - (MAX_AUTO_SAMPLES_PER_STUDENT + PRUNE_BATCH)
    if excess >= 0:
        to_remove = auto_files[:excess + PRUNE_BATCH]
        for old_file in to_remove:
            os.remove(os.path.join(student_dir, old_file))
        gallery[student_id] = retrain_student(student_id, embedder)


def best_match(embedding, gallery):
    """Compares one detected face's embedding against every student's
    bank of embeddings. Returns (best_id, best_sim, runner_up_sim), where
    runner_up_sim is the best score among all OTHER students -- the gap
    between the two is what protects against confusing similar-looking
    people, since a narrow win is ambiguous even if it clears the
    threshold."""
    scores = {}
    for student_id, embeddings in gallery.items():
        scores[student_id] = max(
            cosine_similarity(embedding, gallery_embedding)
            for gallery_embedding in embeddings
        )

    if not scores:
        return None, 0.0, 0.0

    best_id = max(scores, key=scores.get)
    best_sim = scores[best_id]
    runner_up_sim = max((sim for sid, sim in scores.items() if sid != best_id), default=-1.0)
    return best_id, best_sim, runner_up_sim


def find_existing_match(embeddings, gallery):
    """Checks a batch of newly-captured embeddings (from an enrollment in
    progress) against the existing gallery -- used to catch someone who's
    already enrolled trying to register again under a different name.
    Uses the same threshold+margin as real attendance matching, so "is
    this actually the same person" is judged consistently everywhere.
    Returns (student_id, best_similarity) for the strongest hit that
    clears the bar, or (None, 0.0) if nothing does."""
    best_id, best_sim = None, 0.0
    for embedding in embeddings:
        candidate_id, candidate_sim, runner_up_sim = best_match(embedding, gallery)
        margin_ok = (candidate_sim - runner_up_sim) >= MARGIN
        if candidate_id is not None and candidate_sim >= COSINE_THRESHOLD and margin_ok:
            if candidate_sim > best_sim:
                best_id, best_sim = candidate_id, candidate_sim
    return best_id, best_sim


def recognize_faces(frame, aligner, embedder, gallery, detector, antispoof_sessions):
    """Per-photo recognition. Present/Absent for THIS photo is decided
    immediately (any single-photo match counts, same as before -- wrong
    calls here are just corrected next session). Banking is NOT decided
    here: high-confidence matches are returned as candidates only, so the
    caller can require agreement across multiple photos before writing
    anything permanent into the gallery."""
    faces = detect_faces(detector, frame)

    print(f"\nFaces detected in photo: {len(faces)}")

    recognized = {}  # student_id -> best (highest) similarity in this photo
    bank_candidates = []  # (student_id, aligned_face, embedding, similarity) -- not yet banked
    annotated = frame.copy()
    spoofed = 0

    for i, face in enumerate(faces, start=1):
        x, y, w, h = face[:4].astype(int)

        quality_ok, quality_reason = assess_face_quality(frame, (x, y, w, h))
        if not quality_ok:
            box_color = (255, 191, 0)
            text = quality_reason.replace("_", " ").title()
            print(f"  Face {i}: SKIPPED  -> image quality issue ({quality_reason}), not attempting liveness/match")
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            continue

        is_real, real_score = check_liveness(antispoof_sessions, frame, (x, y, w, h))
        if not is_real:
            spoofed += 1
            box_color = (0, 165, 255)
            text = "Spoof"
            print(f"  Face {i}: SPOOF    -> rejected before matching, liveness score={real_score:.3f} (threshold {LIVENESS_THRESHOLD})")
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            continue

        aligned = aligner.alignCrop(frame, face)
        embedding = embed_face(embedder, aligned)

        best_id, best_sim, runner_up_sim = best_match(embedding, gallery)
        margin_ok = (best_sim - runner_up_sim) >= MARGIN

        if best_id is not None and best_sim >= COSINE_THRESHOLD and margin_ok:
            if best_id not in recognized or best_sim > recognized[best_id]:
                recognized[best_id] = best_sim
            box_color = (0, 200, 0)
            text = f"{best_id} ({best_sim:.2f})"
            print(f"  Face {i}: MATCH  -> {best_id:<15} similarity={best_sim:.3f} runner_up={runner_up_sim:.3f} (threshold {COSINE_THRESHOLD}, margin {MARGIN})")

            if best_sim >= AUTO_ENROLL_SIMILARITY and margin_ok:
                bank_candidates.append((best_id, aligned, embedding, best_sim))
                print(f"           bank candidate (similarity >= {AUTO_ENROLL_SIMILARITY}); needs agreement across {MIN_PHOTOS_TO_BANK} photo(s)")
        else:
            box_color = (0, 0, 255)
            text = "Unknown"
            reason = "no gallery" if best_id is None else ("below threshold" if best_sim < COSINE_THRESHOLD else "too close to runner-up")
            print(f"  Face {i}: NO MATCH -> closest guess={best_id!s:<15} similarity={best_sim:.3f} runner_up={runner_up_sim:.3f} ({reason})")

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    return recognized, annotated, bank_candidates, spoofed


def resolve_banking(gallery, all_candidates, num_photos, embedder):
    """Applies the cross-photo agreement rule to everything recognize_faces()
    flagged as bank-worthy across a session, then actually banks the ones
    that qualify. all_candidates: {student_id: [(aligned_face, embedding,
    similarity, photo_idx), ...]}. Returns how many students got a new
    sample banked."""
    required = min(MIN_PHOTOS_TO_BANK, num_photos)
    banked = 0
    for student_id, candidates in all_candidates.items():
        distinct_photos = {photo_idx for _, _, _, photo_idx in candidates}
        if len(distinct_photos) < required:
            print(f"  Not banking {student_id}: matched in only {len(distinct_photos)}/{required} required distinct photo(s).")
            continue
        aligned_face, embedding, similarity, _ = max(candidates, key=lambda c: c[2])
        bank_auto_sample(gallery, student_id, aligned_face, embedding, embedder)
        banked += 1
        print(f"  Banked 1 new sample for {student_id} (agreed across {len(distinct_photos)} photo(s), best similarity={similarity:.3f})")
    if banked > 0:
        save_gallery(gallery)
    return banked


def mark_attendance(recognized_ids):
    students = load_active_roster()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    write_header = not os.path.exists(ATTENDANCE_CSV)
    rows = []
    for student_id, name in sorted(students.items(), key=lambda kv: kv[1]):
        status = "Present" if student_id in recognized_ids else "Absent"
        rows.append([date_str, time_str, student_id, name, status])

    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "time", "student_id", "name", "status"])
        writer.writerows(rows)

    return rows


def main():
    gallery = load_gallery()
    detector = get_face_detector()
    aligner = get_face_aligner()
    embedder = get_embedder()
    antispoof_sessions = get_antispoof_sessions()
    warmup_pipeline(detector, aligner, embedder, antispoof_sessions)

    removed = cleanup_old_captures()
    if removed:
        print(f"Cleaned up {removed} capture photo(s) older than the retention window.")

    frames = capture_photos()
    if not frames:
        print("Cancelled. No photos captured.")
        return

    if not load_active_roster():
        print("No active students in dataset/. Add students first.")
        return

    overall_recognized = {}  # student_id -> best (highest) similarity across all photos
    all_candidates = defaultdict(list)  # student_id -> [(aligned_face, embedding, similarity, photo_idx), ...]
    total_spoofed = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, frame in enumerate(frames, start=1):
        print(f"\n--- Photo {idx}/{len(frames)} ---")
        recognized, annotated, bank_candidates, spoofed = recognize_faces(frame, aligner, embedder, gallery, detector, antispoof_sessions)

        for student_id, similarity in recognized.items():
            if student_id not in overall_recognized or similarity > overall_recognized[student_id]:
                overall_recognized[student_id] = similarity
        for student_id, aligned_face, embedding, similarity in bank_candidates:
            all_candidates[student_id].append((aligned_face, embedding, similarity, idx))
        total_spoofed += spoofed

        snapshot_path = os.path.join(CAPTURES_DIR, f"{timestamp}_{idx}.jpg")
        cv2.imwrite(snapshot_path, annotated)
        print(f"Snapshot saved: {snapshot_path}")

    rows = mark_attendance(overall_recognized)

    print(f"\nAttendance log: {ATTENDANCE_CSV}\n")
    print(f"{'Name':<20} Status")
    print("-" * 30)
    for _, _, _, name, status in rows:
        print(f"{name:<20} {status}")

    if total_spoofed > 0:
        print(f"\nWARNING: {total_spoofed} spoof attempt(s) detected and rejected (photo/screen shown to camera).")

    print()
    total_banked = resolve_banking(gallery, all_candidates, len(frames), embedder)
    if total_banked > 0:
        print(f"\n{total_banked} new high-confidence sample(s) banked.")


if __name__ == "__main__":
    main()
