def best_f1_threshold(scores: list[float], y: list[int]) -> tuple[float, float]:
    """
    The score threshold that maximizes F1 (predict positive if score >= t).
    scores: n predicted scores.
    y: n true labels, each 0 or 1.
    For each distinct value in scores, using it as threshold t, compute F1(t) over the whole dataset.
    Return (best_threshold, best_f1).
    Ties for the best F1 are broken by the LARGER threshold.
    """
    total_positives = sum(y)
    if total_positives == 0 or len(scores) == 0:
        return (0.0, 0.0)

    # Sort descending by score
    sorted_pairs = sorted(zip(scores, y), key=lambda x: x[0], reverse=True)

    running_tp = 0
    running_fp = 0

    best_thresh = float("-inf")
    best_f1 = -1.0

    n = len(sorted_pairs)
    for i in range(n):
        s, label = sorted_pairs[i]
        if label == 1:
            running_tp += 1
        else:
            running_fp += 1

        # Evaluate threshold when at the end of identical scores
        if i == n - 1 or sorted_pairs[i + 1][0] < s:
            t = s
            denom = 2 * running_tp + running_fp + (total_positives - running_tp)
            # denom equals running_tp + running_fp + total_positives
            f1 = (2.0 * running_tp) / (running_tp + running_fp + total_positives) if running_tp > 0 else 0.0

            # Since thresholds are visited in descending order,
            # only strictly greater F1 replaces the previous best to favor the larger threshold on ties.
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

    return (best_thresh, best_f1)
