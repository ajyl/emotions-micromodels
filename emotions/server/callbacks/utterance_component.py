"""
Utility function to update utterance component.
"""
from nltk.tokenize import word_tokenize
from dash import html, dcc, no_update
import dash_bootstrap_components as dbc
from emotions.server.components.analysis import annotated_utterance_component
from emotions.server.callbacks.annotate_utterance import (
    get_annotation_spans,
    annotate_utterance,
)
from emotions.server.utils import entity, get_mm_color
from emotions.constants import (
    THERAPIST,
    MITI_THRESHOLD,
    EMOTION_THRESHOLD,
    EMPATHY_THRESHOLD,
    COG_DIST_THRESHOLD,
)


def build_utterance_tab_component(speaker, active_tab):
    """
    Build utterance_tab component.
    """
    if speaker == THERAPIST:
        return dbc.Tabs(
            [
                dbc.Tab(label="MITI Codes", tab_id="utterance-tab-1"),
                dbc.Tab(label="Emotions", tab_id="utterance-tab-2"),
                dbc.Tab(label="Empathy", tab_id="utterance-tab-3"),
            ],
            id="utterance-tabs",
            active_tab=active_tab,
        )

    return dbc.Tabs(
        [
            dbc.Tab(label="Emotions", tab_id="utterance-tab-2"),
            dbc.Tab(label="Cognitive Distortions", tab_id="utterance-tab-4"),
        ],
        id="utterance-tabs",
        active_tab="utterance-tab-2",
    )


def build_utterance_component(response_obj, speaker, utterance):
    """
    Build utterance component.
    """
    if speaker == THERAPIST:
        default_active_tab = "utterance-tab-1"
        default_mm_type = "miti"
    else:
        default_active_tab = "utterance-tab-2"
        default_mm_type = "custom"

    tab_component = build_utterance_tab_component(speaker, default_active_tab)
    formatted_speaker = speaker[0].upper() + speaker[1:] + ":"

    utterance = " ".join(word_tokenize(utterance))
    utterance_annotation_obj = {
        "utterance": utterance,
        "miti": get_annotation_spans(
            utterance, response_obj, "miti_", MITI_THRESHOLD
        ),
        "custom": get_annotation_spans(
            utterance, response_obj, "custom_", EMOTION_THRESHOLD
        ),
        "empathy": get_annotation_spans(
            utterance, response_obj, "empathy_", EMPATHY_THRESHOLD
        ),
        "cog_dist": get_annotation_spans(
            utterance, response_obj, "cog_dist_", COG_DIST_THRESHOLD
        ),
    }
    annotated_utterance = annotate_utterance(
        utterance_annotation_obj, default_mm_type
    )

    return (
        utterance_annotation_obj,
        [
            tab_component,
            dbc.CardBody(
                [
                    html.H5(formatted_speaker),
                    html.Br(),
                    annotated_utterance_component,
                ],
                id="utterance-card",
            ),
        ],
        annotated_utterance,
    )


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
    ]


def update_utterance_component(annotation_obj, tab_id):
    """
    Update utterance annotation.
    """
    annotation_type = {
        "utterance-tab-1": "miti",
        "utterance-tab-2": "custom",
        "utterance-tab-3": "empathy",
        "utterance-tab-4": "cog_dist",
    }[tab_id]

    return [
        no_update,
        no_update,
        annotate_utterance(annotation_obj, annotation_type),
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
    ]
