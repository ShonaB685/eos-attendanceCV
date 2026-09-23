import os
import re
import shutil
import sys

import cv2

from common import DATASET_DIR, detect_faces, embed_face, get_embedder, get_face_aligner, get_face_detector, load_gallery, save_student
from mark_attendance import find_existing_match
from train_model import retrain_student

SAMPLES_TO_CAPTURE = 25

# Guide the subject through distinct poses instead of firing the shutter
# as fast as detection allows -- back-to-back frames of the same angle
# don't give the recognizer anything new to distinguish this person by.
POSES = [
    "Look straight at the camera",
    "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT",
    "Tilt your chin slightly UP",
    "Tilt your chin slightly DOWN",
]
SAMPLES_PER_POSE = SAMPLES_TO_CAPTURE // len(POSES)
CAPTURE_INTERVAL_MS = 400  # gap between shots within a pose


def sanitize_id(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "student"


def wait_for_ready(cam, headline, captured, target):
    """Show an instruction screen until the subject presses SPACE. Returns False on ESC/cam failure."""
    while True:
        ok, frame = cam.read()
        if not ok:
            return False
        display = frame.copy()
        cv2.putText(display, headline, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, "Press SPACE when ready", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, f"Captured: {captured}/{target}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Add Student - SPACE=ready, ESC=stop", display)

        key = cv2.waitKey(30) & 0xFF
        if key == 32:  # Space
            return True
        if key == 27:  # Esc
            return False


def capture_poses(cam, detector, aligner, student_dir, existing, captured, target, phase_prefix=""):
    """Runs one full pass over POSES, saving aligned crops. Returns (existing, captured, stopped_early)."""
    for pose in POSES:
        pose_label = f"{phase_prefix}{pose}" if phase_prefix else pose
        if not wait_for_ready(cam, pose_label, captured, target):
            return existing, captured, True

        pose_captured = 0
        while pose_captured < SAMPLES_PER_POSE:
            ok, frame = cam.read()
            if not ok:
                return existing, captured, True
            faces = detect_faces(detector, frame)

            display = frame.copy()
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(display, pose_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Captured: {captured}/{target}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(faces) == 0:
                status, status_color = "No face detected - move closer / improve lighting", (0, 0, 255)
            elif len(faces) > 1:
                status, status_color = f"{len(faces)} faces detected - only one person allowed", (0, 0, 255)
            else:
                status, status_color = "Face OK - capturing...", (0, 255, 0)
            cv2.putText(display, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.imshow("Add Student - SPACE=ready, ESC=stop", display)

            if len(faces) == 1:
                aligned = aligner.alignCrop(frame, faces[0])
                captured += 1
                pose_captured += 1
                existing += 1
                cv2.imwrite(os.path.join(student_dir, f"{existing}.jpg"), aligned)
            else:
                print(f"  skipped frame: {len(faces)} face(s) detected (need exactly 1)")

            key = cv2.waitKey(CAPTURE_INTERVAL_MS) & 0xFF
            if key == 27:
                return existing, captured, True

    return existing, captured, False


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = input("Enter student name: ").strip()

    if not name:
        print("Name cannot be empty.")
        sys.exit(1)

    student_id = sys.argv[2] if len(sys.argv) > 2 else sanitize_id(name)

    wears_glasses = input("Does this student wear glasses? (y/n): ").strip().lower().startswith("y")

    student_dir = os.path.join(DATASET_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)
    existing = len([f for f in os.listdir(student_dir) if f.endswith(".jpg")])

    detector = get_face_detector()
    aligner = get_face_aligner()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    # Glasses change the region around the eyes enough that recognition
    # benefits from seeing the subject both with and without them, so
    # glasses-wearers get a second full pose pass -- roughly double the
    # samples of a student who doesn't wear glasses.
    if wears_glasses:
        phases = [
            ("Keep your GLASSES ON - ", "with glasses"),
            ("Remove your glasses now - ", "without glasses"),
        ]
    else:
        phases = [("", None)]

    target = SAMPLES_TO_CAPTURE * len(phases)

    print(f"Enrolling '{name}' (id: {student_id}).")
    print(f"{len(phases)} set(s) x {len(POSES)} poses x {SAMPLES_PER_POSE} samples. "
          "SPACE when ready for each pose, ESC to stop early.")

    captured = 0
    stopped_early = False
    for phase_prefix, phase_note in phases:
        if stopped_early:
            break
        if phase_note:
            print(f"  -- capturing {phase_note} --")
        existing, captured, stopped_early = capture_poses(
            cam, detector, aligner, student_dir, existing, captured, target, phase_prefix
        )

    cam.release()
    cv2.destroyAllWindows()

    if captured == 0:
        print("No face samples captured. Student not added.")
        return

    embedder = get_embedder()
    saved_files = sorted(f for f in os.listdir(student_dir) if f.endswith(".jpg"))
    sample_embeddings = []
    for img_name in saved_files[:10]:  # a sample is enough to check, not all of them
        img = cv2.imread(os.path.join(student_dir, img_name))
        if img is not None:
            sample_embeddings.append(embed_face(embedder, img))

    match_id, match_sim = find_existing_match(sample_embeddings, load_gallery())
    if match_id is not None and match_id != student_id:
        print(f"\nThis face already appears to be enrolled as '{match_id}' (similarity={match_sim:.3f}).")
        print("Not adding as a new/duplicate student -- if this is a mistake, check the existing enrollment.")
        shutil.rmtree(student_dir)
        return

    save_student(student_id, name)
    retrain_student(student_id, embedder)
    print(f"Captured {captured} samples for '{name}'. Gallery updated.")


if __name__ == "__main__":
    main()
