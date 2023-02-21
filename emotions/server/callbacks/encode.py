"""
Callback methods
"""

import json
import requests
from nltk.tokenize import word_tokenize
from dash import html, callback, Input, Output, State, ALL, ctx, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from emotions.constants import (
    FEATURIZER_SERVER,
    THERAPIST,
)
from emotions.server.components.analysis import (
    utterance_component,
    annotated_utterance_component,
    micromodel_bar_graph,
    explanation_graph,
    emotion_table,
)
from emotions.server.components import (
    current_search_idx,
)
from emotions.server.callbacks import (
    build_emotion_analysis_component,
    build_empathy_analysis_component,
    build_utterance_component,
    build_micromodel_component,
    build_explanation_component,
    update_utterance_component,
    update_global_exp,
    handle_hover,
)
from emotions.server.utils import entity, get_mm_color


@callback(
    [
        Output({"type": "popover", "index": ALL}, "is_open"),
        Output({"type": "popover_body", "index": ALL}, "children"),
        Output("dialogue_idx", "data"),
    ],
    [
        Input({"type": "dialogue-click", "index": ALL}, "n_clicks"),
        Input("conversation-encoding", "data"),
        Input("micromodel-results", "hoverData"),
        Input("global-explanation-feature-dropdown", "value"),
        Input("utterance-tabs", "active_tab"),
        Input("annotated-utterance-storage", "data"),
    ],
    [
        State("dialogue_idx", "data"),
        State({"type": "popover", "index": ALL}, "is_open"),
        State({"type": "dialogue-click", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def analysis_popup(
    n_clicks,
    conversation_encoding,
    hover_data,
    explanation_dropdown,
    active_utterance_tab,
    annotated_utterance_storage,
    prev_idx,
    is_open,
    utterances,
):
    triggered_id = ctx.triggered_id
    print(triggered_id)
    print("n_clicks", n_clicks)
    if sum(n_clicks) == 0:
        raise PreventUpdate

    if triggered_id == "conversation-encoding":
        raise PreventUpdate

    if triggered_id == "global-explanation-feature-dropdown":
        breakpoint()

    if triggered_id in ["utterance-tabs", "micromodel-results"]:
       #return update_utterance_component(
       #    annotated_utterance_storage, active_utterance_tab
       #)
       idx = prev_idx
       print(active_utterance_tab)

    else:
        idx = ctx.triggered_id["index"]



    _is_open = [False] * len(is_open)
    _is_open[idx] = True

    utterance_obj = utterances[idx]
    utterance = utterance_obj["utterance"]
    speaker = utterance_obj["speaker"]
    utterance_encoding = conversation_encoding[idx]["results"]

    (
        tab_component,
        formatted_speaker,
        annotated_utterance,
    ) = build_utterance_component(
        utterance_obj, utterance_encoding, active_utterance_tab
    )

    if triggered_id == "micromodel-results":
        data = hover_data["points"][0]
        micromodel = data["label"]
        query = " ".join(word_tokenize(data["customdata"][1]))
        segment = " ".join(word_tokenize(data["customdata"][0]))
    
        if segment == "":
            raise PreventUpdate

        else:
            segment_start_idx = query.index(segment)
            segment_end_idx = segment_start_idx + len(segment)
    
            annotated_utterance = (
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

    micromodel_component = build_micromodel_component(
        utterance_encoding["micromodels"], utterance, speaker
    )

    # Emotions - Predictions
    emotion_classifications = utterance_encoding["emotion"]["predictions"][0][
        0
    ]

    # Emotions - Explanations
    explanation_fig = build_explanation_component(
        explanation_dropdown, emotion_classifications
    )

    emotion_analysis_table = build_emotion_analysis_component(
        utterance_encoding["emotion"]["predictions"], explanation_dropdown
    )

    analysis = html.Div(
        children=[
            html.Div(
                [
                    dbc.Card(
                        [
                            tab_component,
                            dbc.CardBody(
                                [
                                    html.H5(formatted_speaker),
                                    html.Br(),
                                    html.Div(annotated_utterance),
                                ]
                            ),
                        ]
                    ),
                    html.Br(),
                    micromodel_component,
                    html.Br(),
                    emotion_analysis_table,
                ],
                style={"display": "block"},
            ),
        ],
    )

    popover = [{}] * len(is_open)
    popover[idx] = analysis
    return [
        _is_open,
        popover,
        idx,
    ]
