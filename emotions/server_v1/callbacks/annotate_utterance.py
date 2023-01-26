"""
Functionality around annotating utterances.
"""

from nltk.tokenize import word_tokenize
from dash import no_update
from emotions.server.utils import get_mm_color, entity


def annotate_utterance(annotation_obj, annotation_type):
    """
    Return an annotated utterance based on :annotation_obj:.
    """
    utterance = annotation_obj["utterance"]
    annotation_idxs = annotation_obj[annotation_type]
    annotated_utterance = utterance
    if len(annotation_idxs) > 0:
        annotated_utterance = []
        last_idx = 0
        for idx_obj in annotation_idxs:
            # idx_obj[0]: span label
            # idx_obj[1]: span start idx
            # idx_obj[2]: span end idx
            # idx_obj[3]: span score
            annotated_utterance.extend(utterance[last_idx : idx_obj[1]])
            annotated_utterance.append(
                entity(
                    utterance[idx_obj[1] : idx_obj[2]],
                    idx_obj[0],
                    get_mm_color(annotation_type),
                )
            )
            last_idx = idx_obj[2]
        annotated_utterance.extend(utterance[last_idx:])
    return annotated_utterance


def get_annotation_spans(utterance, response_obj, prefix, threshold):
    """
    Get emotion spans.
    """
    mm_results = [
        (x, y["max_score"])
        for x, y in response_obj["micromodels"].items()
        if x.startswith(prefix) and y["max_score"] >= threshold
    ]
    mm_results = sorted(
        mm_results,
        key=lambda x: x[1],
        reverse=True,
    )
    segments = [
        (
            mm_type,
            " ".join(
                word_tokenize(
                    response_obj["micromodels"][mm_type]["segment"]
                )
            ),
        )
        for mm_type, _ in mm_results
    ]
    segment_idxs = [
        (
            emotion.replace("custom_", ""),
            utterance.index(_segment),
            utterance.index(_segment) + len(_segment),
            response_obj["micromodels"][emotion]["max_score"],
        )
        for emotion, _segment in segments
    ]
    segment_idxs = sorted(segment_idxs, key=lambda x: x[1])
    if len(segment_idxs) < 1:
        return segment_idxs

    merged_spans = []

    prev = segment_idxs[0]
    for idx_obj in segment_idxs[1:]:
        curr_min = prev[1]
        curr_max = prev[2]
        curr_score = prev[3]

        new_min = idx_obj[1]
        new_score = idx_obj[3]

        # Overlap
        if curr_min <= new_min <= curr_max:
            if new_score > curr_score:
                prev = idx_obj

        # No Overlap
        elif new_min > curr_max:
            merged_spans.append(prev)
            prev = idx_obj

        else:
            # Should never reach here.
            breakpoint()

    if prev not in merged_spans:
        merged_spans.append(prev)

    return merged_spans
