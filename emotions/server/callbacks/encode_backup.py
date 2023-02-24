"""
Callback methods
"""

import json
import requests
from dash import callback, Input, Output, State, ALL, ctx, no_update
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
    empathy_table,
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


@callback(
    [
        Output("analysis", "style"),
        Output(utterance_component, "children"),
        Output(annotated_utterance_component, "children"),
        Output(micromodel_bar_graph, "children"),
        Output(explanation_graph, "children"),
        Output(emotion_table, "children"),
        Output(empathy_table, "children"),
        Output("annotated-utterance-storage", "data"),
        Output("emotion-classification-storage", "data"),
    ],
    [
        Input("conversation-encoding", "data"),
        Input("micromodel-results", "hoverData"),
        Input("global-explanation-feature-dropdown", "value"),
        Input("emotion-classification-storage", "data"),
        Input("utterance-tabs", "active_tab"),
        Input("annotated-utterance-storage", "data"),
        Input("current-result-idx-storage", "data"),
        Input({"type": "dialogue-click", "index": ALL}, "n_clicks"),
    ],
    [
        State({"type": "dialogue-click", "index": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def encode(
    conversation_encoding,
    hover_data,
    explanation_dropdown,
    emotion_classification_storage,
    active_utterance_tab,
    annotated_utterance_storage,
    search_idx,
    n_clicks,
    utterances,
):

    triggered_id = ctx.triggered_id

    #if sum(n_clicks) == 0:
    #    return [
    #        {"display": "none"},
    #        no_update,
    #        no_update,
    #        no_update,
    #        no_update,
    #        no_update,
    #        no_update,
    #        no_update,
    #        no_update,
    #    ]
    if triggered_id is None:
        raise PreventUpdate

    if triggered_id == "conversation-encoding":
        raise PreventUpdate

    if triggered_id == "micromodel-results":
        return handle_hover(hover_data)

    if triggered_id == "utterance-tabs":
        return update_utterance_component(
            annotated_utterance_storage, active_utterance_tab
        )

    if triggered_id == "global-explanation-feature-dropdown":
        return update_global_exp(
            explanation_dropdown,
            emotion_classification_storage,
        )

    if triggered_id == "current-result-idx-storage":
        idx = search_idx
    else:
        idx = ctx.triggered_id["index"]

    utterance_obj = utterances[idx]
    utterance = utterance_obj["utterance"]
    speaker = utterance_obj["speaker"]
    utterance_encoding = conversation_encoding[idx]["results"]

    (
        annotated_utterance_obj,
        _utterance_component,
        annotated_utterance,
    ) = build_utterance_component(utterance_encoding, speaker, utterance)

    micromodel_component = build_micromodel_component(
        utterance_encoding["micromodels"], utterance, speaker
    )

    # Emotions - Predictions
    emotion_classifications = utterance_encoding["emotion"]["predictions"][0][0]

    # Emotions - Explanations
    explanation_fig = build_explanation_component(
        explanation_dropdown, emotion_classifications
    )

    emotion_analysis_table = build_emotion_analysis_component(
        utterance_encoding["emotion"]["predictions"], explanation_dropdown
    )
    empathy_analysis_table = build_empathy_analysis_component(
        utterance_encoding["micromodels"]
    )

    return [
        {"display": "block"},
        _utterance_component,
        annotated_utterance,
        micromodel_component,
        explanation_fig,
        emotion_analysis_table,
        empathy_analysis_table,
        annotated_utterance_obj,
        emotion_classifications,
    ]
