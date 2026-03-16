from rapidfuzz import fuzz


def fuzzy_match(
    candidate_value: str | None,
    user_value: str,
    threshold: float = 0.7,
) -> tuple[bool, float]:
    """Check if a candidate field value matches the user's answer.

    Uses token_sort_ratio for order-independent matching.
    Handles common abbreviations and variations.

    Args:
        candidate_value: Value from the candidate profile (may be None).
        user_value: Value provided by the user.
        threshold: Similarity threshold 0.0-1.0.

    Returns:
        Tuple of (is_match: bool, similarity_score: float).
    """
    if candidate_value is None:
        return False, 0.0

    # Normalize
    cv = candidate_value.strip().lower()
    uv = user_value.strip().lower()

    if not cv or not uv:
        return False, 0.0

    # Exact match shortcut
    if cv == uv:
        return True, 1.0

    # Containment check (e.g., "University of Oregon" contains "Oregon")
    if uv in cv or cv in uv:
        return True, 0.9

    # Fuzzy match with token sort (handles word order differences)
    score = fuzz.token_sort_ratio(cv, uv) / 100.0

    return score >= threshold, score
