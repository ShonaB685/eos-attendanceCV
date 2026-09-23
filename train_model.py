import os

import cv2

from common import DATASET_DIR, embed_face, get_embedder, load_gallery, save_gallery


def train(verbose=True):
    """Rebuild the embedding gallery from everything currently in dataset/.
    Each dataset/<id>/*.jpg is expected to already be an aligned 112x112
    face crop (that's what add_student.py and the auto-enroll banking
    produce). Returns True on success, False if there was nothing usable.
    This re-embeds EVERY student's images from scratch -- slow (scales
    with total dataset size), so use retrain_student() instead for "just
    add this one new student" (enrollment) or "just refresh this one
    student's samples" (pruning) rather than calling this on every
    enrollment."""
    student_ids = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not student_ids:
        if verbose:
            print("No students found in dataset/. Run add_student.py first.")
        return False

    embedder = get_embedder()
    gallery = {}

    for student_id in student_ids:
        embeddings = _embed_student_images(student_id, embedder)
        if embeddings:
            gallery[student_id] = embeddings
        if verbose:
            print(f"  {student_id}: {len(embeddings)} samples")

    if not gallery:
        if verbose:
            print("No usable face images found.")
        return False

    save_gallery(gallery)

    if verbose:
        total = sum(len(v) for v in gallery.values())
        print(f"Built gallery from {total} images across {len(gallery)} students.")

    return True


def _embed_student_images(student_id, embedder):
    student_dir = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(student_dir):
        return []
    images = [f for f in os.listdir(student_dir) if f.lower().endswith(".jpg")]
    embeddings = []
    for img_name in images:
        img = cv2.imread(os.path.join(student_dir, img_name))
        if img is None:
            continue
        embeddings.append(embed_face(embedder, img))
    return embeddings


def retrain_student(student_id, embedder=None):
    """Re-embeds just ONE student's current images and replaces their
    portion of the gallery -- O(that student's own image count), not the
    whole dataset. This is what enrollment and auto-bank pruning should
    call; a full train() re-embeds everyone and scales with total dataset
    size (measured ~154s at 275 images across 8 students -- unusable to
    run on every single enrollment)."""
    embedder = embedder or get_embedder()
    embeddings = _embed_student_images(student_id, embedder)

    gallery = load_gallery()
    if embeddings:
        gallery[student_id] = embeddings
    else:
        gallery.pop(student_id, None)
    save_gallery(gallery)
    return embeddings


if __name__ == "__main__":
    train()
