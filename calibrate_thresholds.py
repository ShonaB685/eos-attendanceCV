"""Measures actual FAR/FRR for COSINE_THRESHOLD/MARGIN against the real
enrolled gallery, via leave-one-out 1:N identification -- replaces the
placeholder defaults with values backed by real data instead of a guess.
Needs a reasonable number of enrolled students/samples to mean anything;
re-run as more real people and more real-world auto-banked samples get
enrolled, since the recommendation will keep shifting."""

import numpy as np

from common import cosine_similarity, load_gallery


def leave_one_out_scores():
    """For every embedding, simulates best_match() with itself excluded
    from its own student's bank (so it can't trivially "match itself").
    Returns a list of (true_student, best_id, best_sim, runner_up_sim)
    tuples -- one per embedding in the gallery."""
    gallery = load_gallery()
    results = []

    for true_student, embeddings in gallery.items():
        for i, query in enumerate(embeddings):
            scores = {}
            for student_id, student_embeddings in gallery.items():
                candidates = [
                    e for j, e in enumerate(student_embeddings)
                    if not (student_id == true_student and j == i)
                ]
                if not candidates:
                    continue
                scores[student_id] = max(cosine_similarity(query, e) for e in candidates)

            if not scores:
                continue
            best_id = max(scores, key=scores.get)
            best_sim = scores[best_id]
            runner_up_sim = max((s for sid, s in scores.items() if sid != best_id), default=-1.0)
            results.append((true_student, best_id, best_sim, runner_up_sim))

    return results


def evaluate(results, threshold, margin):
    """Applies the real accept rule (best_sim >= threshold AND
    best_sim - runner_up_sim >= margin) to every leave-one-out query.
    Returns (far, frr): FAR is impostor queries wrongly ACCEPTED as some
    other real student; FRR is genuine queries wrongly REJECTED."""
    genuine_total = 0
    genuine_rejected = 0
    impostor_total = 0
    impostor_accepted = 0

    for true_student, best_id, best_sim, runner_up_sim in results:
        accept = best_sim >= threshold and (best_sim - runner_up_sim) >= margin
        if best_id == true_student:
            genuine_total += 1
            if not accept:
                genuine_rejected += 1
        else:
            impostor_total += 1
            if accept:
                impostor_accepted += 1

    frr = genuine_rejected / genuine_total if genuine_total else 0.0
    far = impostor_accepted / impostor_total if impostor_total else 0.0
    return far, frr


def main():
    print("Running leave-one-out identification over the current gallery...\n")
    results = leave_one_out_scores()
    if not results:
        print("No embeddings in the gallery -- enroll students first.")
        return

    students = sorted({r[0] for r in results})
    print(f"{len(results)} query embeddings across {len(students)} students: {', '.join(students)}\n")

    genuine_count = sum(1 for r in results if r[0] == r[1])
    impostor_count = len(results) - genuine_count
    print(f"Baseline (argmax alone, no threshold/margin): {genuine_count} naturally match the right "
          f"student, {impostor_count} naturally match the WRONG one.")
    if impostor_count == 0:
        print("(0 misidentifications at raw argmax means FAR will read 0% at every threshold below --")
        print(" that's not proof the threshold is safe in general, just that these particular students")
        print(" aren't confusable with each other. FRR is the only signal this run can actually give you.)")
    print()

    print(f"{'threshold':>10} {'margin':>8} {'FAR':>8} {'FRR':>8}")
    current_threshold, current_margin = 0.40, 0.07
    candidates = []
    for threshold in np.arange(0.0, 0.71, 0.05):
        for margin in (0.0, 0.03, 0.05, 0.07, 0.10):
            far, frr = evaluate(results, threshold, margin)
            print(f"{threshold:>10.2f} {margin:>8.2f} {far:>8.1%} {frr:>8.1%}")
            candidates.append((far + frr, threshold, margin, far, frr))

    # Among every (threshold, margin) tied for lowest combined error, prefer
    # whichever sits closest to the current defaults -- picking the extreme
    # edge (e.g. threshold=0.0) of a zero-error zone isn't a meaningful
    # recommendation, it just means the sweep didn't find real risk there.
    min_score = min(c[0] for c in candidates)
    tied = [c for c in candidates if c[0] == min_score]
    best = min(tied, key=lambda c: abs(c[1] - current_threshold) + abs(c[2] - current_margin))

    print(f"\n{len(tied)} threshold/margin combination(s) tied for lowest combined error ({min_score:.1%}).")
    print(f"Closest to current defaults (threshold={current_threshold}, margin={current_margin}): "
          f"threshold={best[1]:.2f}, margin={best[2]:.2f} (FAR={best[3]:.1%}, FRR={best[4]:.1%})")
    print("Starting recommendation, not gospel -- weigh FAR vs FRR by what matters more for your use "
          "case (a stricter threshold trades some legitimate rejections for fewer wrong-identity accepts),")
    print("and re-run this as more/more-diverse students get enrolled -- 8 students who don't happen to")
    print("resemble each other is a small, easy case for the recognizer.")


if __name__ == "__main__":
    main()
