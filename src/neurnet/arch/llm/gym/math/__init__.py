from __future__ import annotations

from .parse import equality_check, normalize_text, split_into_parts


def grade_answer(pred_text: str | None, gt_text: str | None) -> bool:
    result = False  # Default outcome if checks fail

    # Only continue if both inputs are non-empty strings
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )  # Break ground truth into comparable parts

        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )  # Break prediction into comparable parts

        # Ensure both sides have same number of valid parts
        if gt_parts and pred_parts and len(gt_parts) == len(pred_parts):
            result = all(
                equality_check(gt, pred) for gt, pred in zip(gt_parts, pred_parts)
            )  # Check each part for mathematical equivalence

    return result  # True only if all checks passed
