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
    utterance_tabs_card,
    utterance_tabs,
    utterance_component_container,
    annotated_utterance_component,
    micromodel_bar_graph,
    micromodel_bar_graph_container,
    explanation_graph,
    emotion_table,
)
from emotions.server.components import (
    current_search_idx,
)
from emotions.server.callbacks import (
    # build_emotion_analysis_component,
    # build_empathy_analysis_component,
    build_utterance_component,
    #build_micromodel_component,
    # build_explanation_component,
    update_utterance_component,
    # update_global_exp,
    handle_hover,
)
from emotions.server.utils import entity, get_mm_color


@callback(
    [
        Output({"type": "popover", "index": ALL}, "is_open"),
        # Output({"type": "popover_body", "index": ALL}, "children"),
        Output("dialogue_idx", "data"),
        Output(
            {"type": "annotated-utterance-container", "index": ALL}, "children"
        ),
        # Output(micromodel_bar_graph_container, "children"),
        # Output(utterance_tabs_card, "children"),
        # Output(utterance_component_container, "children"),
    ],
    [
        Input({"type": "dialogue-click", "index": ALL}, "n_clicks"),
        Input("conversation-encoding", "data"),
        # Input("micromodel-results", "hoverData"),
        Input({"type": "micromodel-results", "index": ALL}, "hoverData"),
        # Input("global-explanation-feature-dropdown", "value"),
        # Input(utterance_tabs, "active_tab"),
        Input({"type": "utterance-tabs", "index": ALL}, "active_tab"),
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
    hover_datas,
    active_tabs,
    # explanation_dropdown,
    # active_utterance_tab,
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

    if triggered_id.get("type") ==  "micromodel-results":
        idx = prev_idx

        if hover_datas is None:
            raise PreventUpdate

        hover_data = hover_datas[idx]
        data = hover_data["points"][0]
        micromodel = data["label"]
        query = " ".join(word_tokenize(data["customdata"][1]))
        segment = " ".join(word_tokenize(data["customdata"][0]))

        if segment == "":
            raise PreventUpdate

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
        annotated_utterances = [no_update] * len(is_open)
        annotated_utterances[idx] = annotated_utterance

        _is_open = [False] * len(is_open)
        _is_open[idx] = True

        return [_is_open, idx, annotated_utterances]

    elif triggered_id.get("type") == "utterance-tabs":
        print("active_tab", active_tabs)
        idx = ctx.triggered_id["index"]
        active_tab = active_tabs[idx]

        utterance_obj = utterances[idx]
        utterance = utterance_obj["utterance"]
        speaker = utterance_obj["speaker"]
        utterance_encoding = conversation_encoding[idx]["results"]

        (_, _, annotated_utterance,) = build_utterance_component(
            utterance_obj, utterance_encoding, active_tab, idx
        )
        annotated_utterances = [no_update] * len(is_open)
        annotated_utterances[idx] = annotated_utterance

        _is_open = [False] * len(is_open)
        _is_open[idx] = True
        return [_is_open, idx, annotated_utterances]

    else:
        idx = ctx.triggered_id["index"]

        if idx == prev_idx:

            popover = [{}] * len(is_open)
            _is_open = [False] * len(is_open)
            # return [_is_open, popover, idx, no_update, no_update, no_update]
            # return [_is_open, popover, idx, no_update, no_update]
            return [_is_open, idx, [no_update] * len(is_open)]

    _is_open = [False] * len(is_open)
    _is_open[idx] = True

    # utterance_obj = utterances[idx]
    # utterance = utterance_obj["utterance"]
    # speaker = utterance_obj["speaker"]
    # utterance_encoding = conversation_encoding[idx]["results"]

    # (
    #    tab_component,
    #    formatted_speaker,
    #    annotated_utterance,
    # ) = build_utterance_component(
    #    utterance_obj, utterance_encoding, utterance_tabs_component
    # )

    # if triggered_id == "micromodel-results":
    #    if hover_data is None:
    #        raise PreventUpdate
    #    data = hover_data["points"][0]
    #    micromodel = data["label"]
    #    query = " ".join(word_tokenize(data["customdata"][1]))
    #    segment = " ".join(word_tokenize(data["customdata"][0]))

    #    if segment == "":
    #        raise PreventUpdate

    #    else:
    #        segment_start_idx = query.index(segment)
    #        segment_end_idx = segment_start_idx + len(segment)

    #        annotated_utterance = (
    #            [query[:segment_start_idx]]
    #            + [
    #                entity(
    #                    query[segment_start_idx:segment_end_idx],
    #                    micromodel,
    #                    get_mm_color(micromodel),
    #                )
    #            ]
    #            + [query[segment_end_idx:]]
    #        )

    # micromodel_component = build_micromodel_component(
    #    utterance_encoding["micromodels"], utterance, speaker
    # )

    # _utterance_component = dbc.Card(
    #    [
    #        dbc.CardHeader(tab_component),
    #        dbc.CardBody(
    #            [
    #                html.H5(formatted_speaker),
    #                html.Br(),
    #                html.Div(annotated_utterance),
    #            ]
    #        ),
    #    ]
    # )

    # analysis = html.Div(
    #    children=[
    #        html.Div(
    #            [
    #                _utterance_component,
    #                html.Br(),
    #                micromodel_component,
    #            ],
    #            style={"display": "block"},
    #        ),
    #    ],
    # )

    # popover = [{}] * len(is_open)
    # popover[idx] = analysis
    return [
        _is_open,
        # popover,
        idx,
        [no_update] * len(is_open)
        # micromodel_component,
        # tab_component,
        # _utterance_component
    ]
