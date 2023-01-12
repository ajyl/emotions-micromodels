"""
Functionality around hovering.
"""

from nltk.tokenize import word_tokenize
from dash import no_update
from emotions.server.utils import entity, get_mm_color


def handle_hover(hover_data):
    """
    Handle hovering over micromodel vector.
    """
    data = hover_data["points"][0]
    micromodel = data["label"]
    query = " ".join(word_tokenize(data["customdata"][1]))
    segment = " ".join(word_tokenize(data["customdata"][0]))

    if segment == "":
        annotated_query = no_update
    else:
        segment_start_idx = query.index(segment)
        segment_end_idx = segment_start_idx + len(segment)

        annotated_query = (
            [query[:segment_start_idx]]
            + [
                entity(
                    query[segment_start_idx:segment_end_idx],
                    micromodel,
                    get_mm_color(micromodel),
                )
            ]
            + [query[segment_end_idx:]]
        )
    return [
        no_update,
        no_update,
        annotated_query,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
    ]
