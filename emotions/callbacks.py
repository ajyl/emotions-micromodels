"""
Callback methods
"""

import json
import pickle
import base64
import requests
import pandas as pd
from nltk.tokenize import word_tokenize
from dash import html, dcc, callback, Input, Output, State, ALL, ctx, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px

from constants import FEATURIZER_SERVER, MITI_THRESHOLD

COLOR_SCHEME = px.colors.qualitative.Set2


def _get_mm_color(mm_name):
    """
    Get color for mm.
    """
    mm_prefixes = ["emotion_", "custom_", "empathy_", "miti_"]
    for idx, prefix in enumerate(mm_prefixes):
        if mm_name.startswith(prefix):
            return COLOR_SCHEME[idx]
    raise ValueError("Unknown MM %s!" % mm_name)


def entname(name):
    return html.Span(
        name,
        style={
            "font-size": "0.8em",
            "font-weight": "bold",
            "line-height": "1",
            "border-radius": "0.35em",
            "text-transform": "uppercase",
            "vertical-align": "middle",
            "margin-left": "0.5rem",
            "margin-right": "0.5rem",
        },
    )


def entbox(children, color):
    return html.Mark(
        children,
        style={
            "background": color,
            "padding": "0.45em 0.6em",
            "margin": "0 0.25em",
            "line-height": "1",
            "border-radius": "0.35em",
        },
    )


def entity(text, entity_name, color):
    assert isinstance(text, str)
    text = [entname(entity_name)] + [text]
    return entbox(text, color)


def handle_hover(hover_data):
    """
    Handle hovering over micromodel vector.
    """
    data = hover_data["points"][0]
    micromodel = data["label"]
    query = " ".join(word_tokenize(data["customdata"][3]))
    segment = " ".join(word_tokenize(data["customdata"][2]))

    segment_start_idx = query.index(segment)
    segment_end_idx = segment_start_idx + len(segment)

    annotated_query = (
        [query[:segment_start_idx]]
        + [
            entity(
                query[segment_start_idx:segment_end_idx],
                micromodel,
                _get_mm_color(micromodel),
            )
        ]
        + [query[segment_end_idx:]]
    )
    # for idx, char in enumerate(query):
    #    if segment_start_idx <= idx <= segment_end_idx:
    #        annotated_query.append(
    #            html.Span(
    #                char, style={"background-color": _get_mm_color(micromodel)}
    #            )
    #        )
    #    else:
    #        annotated_query.append(char)

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
    ]


def update_global_exp(
    global_exp_storage, explanation_dropdown, emotion_classifications
):
    """
    Update global explanation figure.
    """
    global_exp = pickle.loads(base64.b64decode(global_exp_storage))
    visual_idx = None
    if explanation_dropdown in global_exp.feature_names:
        visual_idx = global_exp.feature_names.index(explanation_dropdown)

    global_fig = global_exp.visualize(visual_idx)
    global_fig.update_layout(legend_title="Predictions:")

    if visual_idx is not None:
        global_fig["data"] = global_fig["data"][:-1]
        orig_idxs = {_idx: x for _idx, x in enumerate(global_fig["data"])}
        orig_order = [x["name"] for x in global_fig["data"]]
        global_fig["data"] = [
            orig_idxs[orig_order.index(emotion)]
            for emotion in emotion_classifications
        ]
        global_fig["layout"].pop("xaxis")
        global_fig["layout"].pop("yaxis")

    return [
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
        global_fig,
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


@callback(
    [
        Output("analysis", "style"),
        Output("speaker", "children"),
        Output("utterance", "children"),
        Output("micromodel-results", "figure"),
        Output("micromodel-results", "style"),
        Output("global-explanation", "figure"),
        Output("global-explanation", "style"),
        Output("global-explanation-storage", "data"),
        Output("emotion-classification-storage", "data"),
        Output("emotion_1", "children"),
        Output("emotion_score_1", "children"),
        Output("emotion_2", "children"),
        Output("emotion_score_2", "children"),
        Output("emotion_3", "children"),
        Output("emotion_score_3", "children"),
        Output("emotion-classification-table", "style"),
        Output("empathy", "children"),
    ],
    [
        Input({"type": "dialogue-click", "index": ALL}, "n_clicks"),
        Input("micromodel-results", "hoverData"),
        Input("global-explanation-feature-dropdown", "value"),
        Input("global-explanation-storage", "data"),
        Input("emotion-classification-storage", "data"),
    ],
    [
        State({"type": "dialogue-click", "index": ALL}, "value"),
    ],
)
def encode(
    n_clicks,
    hover_data,
    explanation_dropdown,
    explanation_storage,
    emotion_classification_storage,
    utterances,
):

    print("Explanation_dropdown")
    print(explanation_dropdown)
    if sum(n_clicks) == 0:
        return [
            {"display": "none"},
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

    triggered_inputs = ctx.triggered
    hovered = any(
        x.get("prop_id") == "micromodel-results.hoverData"
        for x in triggered_inputs
    )
    if hovered:
        return handle_hover(hover_data)

    if ctx.triggered_id is None:
        raise PreventUpdate

    triggered_id = ctx.triggered_id

    # TODO: Is there a way to avoid another network call
    # for this case?
    if triggered_id == "global-explanation-feature-dropdown":
        return update_global_exp(
            explanation_storage,
            explanation_dropdown,
            emotion_classification_storage,
        )

    idx = ctx.triggered_id["index"]
    utterance_obj = utterances[idx]
    utterance = utterance_obj["utterance"]

    speaker = utterance_obj["speaker"]
    response = requests.post(
        FEATURIZER_SERVER + "/encode", json={"query": utterance}
    )
    result = response.json()

    # Classifications
    classifications = result["classifications"]
    emotion_classifications = classifications["emotions"][0][0]

    empathy = [
        classifications["empathy_emotional_reactions"],
        classifications["empathy_explorations"],
        classifications["empathy_interpretations"],
    ]

    # Explanations
    explanations = result["explanations"]
    global_exp = pickle.loads(base64.b64decode(explanations["global"]))
    visual_idx = None
    if explanation_dropdown in global_exp.feature_names:
        visual_idx = global_exp.feature_names.index(explanation_dropdown)

    global_fig = global_exp.visualize(visual_idx)
    global_fig.update_layout(legend_title="Predictions:")

    if visual_idx is not None:
        global_fig["data"] = global_fig["data"][:-1]
        orig_idxs = {_idx: x for _idx, x in enumerate(global_fig["data"])}
        orig_order = [x["name"] for x in global_fig["data"]]
        global_fig["data"] = [
            orig_idxs[orig_order.index(emotion)]
            for emotion in emotion_classifications
        ]
        global_fig["layout"].pop("xaxis")
        global_fig["layout"].pop("yaxis")

    # MITI
    miti_results = {
        x: y for x, y in result["results"].items() if x.startswith("miti_")
    }
    _miti = sorted(
        [
            (x, y["max_score"])
            for x, y in miti_results.items()
            if y["max_score"] >= MITI_THRESHOLD
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    segments = [
        (miti_code[0], result["results"][miti_code[0]]["segment"])
        for miti_code in _miti
    ]
    # TODO
    # if len(_miti) > 0:
    #    utterance = " ".join(word_tokenize(utterance_obj["utterance"]))
    #    for _miti_code, _segment in segments:
    #        _idx = utterance.index(_segment)
    #        utterance = (
    #            utterance[:_idx]
    #            + "[[ %s %s ]]"
    #            % (_miti_code.upper().replace("_", "-"), _segment)
    #            + utterance[_idx + len(_segment) :]
    #        )

    # Micromodels
    mms = list(result["results"].keys())
    sorted_mms = (
        sorted([mm for mm in mms if mm.startswith("emotion_")])
        + sorted([mm for mm in mms if mm.startswith("custom_")])
        + sorted([mm for mm in mms if mm.startswith("empathy_")])
        + sorted([mm for mm in mms if mm.startswith("miti_")])
    )

    data = []
    for mm in sorted_mms:
        mm_result = result["results"][mm]
        data.append(
            (
                mm,
                utterance,
                max(mm_result["max_score"], 0),
                mm_result["top_k_scores"][0][0],
                mm_result["top_k_scores"][0][1],
                mm_result["segment"],
                _get_mm_color(mm),
            )
        )

    data = pd.DataFrame(
        data=data,
        columns=[
            "Micromodel",
            "query",
            "score",
            "similar_segment",
            "similar_score",
            "segment",
            "color",
        ],
    )
    fig = px.bar(
        data_frame=data,
        x="score",
        y="Micromodel",
        color="color",
        hover_name="Micromodel",
        hover_data=["Micromodel", "score", "similar_segment", "similar_score"],
        custom_data=["similar_segment", "similar_score", "segment", "query"],
        orientation="h",
        height=1200,
        color_discrete_sequence=COLOR_SCHEME,
    )
    fig.update_coloraxes(showscale=False)
    fig.layout.showlegend = False
    fig.update_layout()

    speaker = speaker[0].upper() + speaker[1:] + ":"

    miti_segments = [
        (
            miti_code[0],
            " ".join(
                word_tokenize(result["results"][miti_code[0]]["segment"])
            ),
        )
        for miti_code in _miti
    ]

    custom_emotion_segments = [
        (x, " ".join(word_tokenize(result["results"][x]["segment"])))
        for x in sorted_mms
        if x.startswith("custom_")
    ]

    _query = " ".join(word_tokenize(utterance))
    miti_idxs = [
        (
            miti_code,
            _query.index(_segment),
            _query.index(_segment) + len(_segment),
        )
        for miti_code, _segment in miti_segments
    ]
    custom_emotion_idxs = [
        (_query.index(_segment), _query.index(_segment) + len(_segment))
        for _, _segment in custom_emotion_segments
    ]

    miti_idxs = sorted(miti_idxs, key=lambda x: x[1])

    miti_idx = 0

    annotated_utterance = _query
    if len(miti_idxs) > 0:
        annotated_utterance = []
        for miti_idx in miti_idxs:
            annotated_utterance.extend(_query[: miti_idx[1]])
            annotated_utterance.append(
                entity(
                    _query[miti_idx[1] : miti_idx[2]],
                    miti_idx[0],
                    _get_mm_color("miti_"),
                )
            )
        annotated_utterance.extend(_query[miti_idx[2] :])

    # for idx, char in enumerate(_query):
    #    curr_miti = d

    #    annotated_

    #    miti_spans = []
    #    for miti_range in miti_idxs:
    #        if idx >= miti_range[1] and idx <= miti_range[2]:
    #            miti_spans.append(miti_range[0])

    #    if len(miti_spans) > 0:
    #        annotated_utterance.append(
    #            html.Span(
    #                char,
    #                style={"background-color": "#90EE90", "border": "2px"},
    #            )
    #        )
    #    else:
    #        annotated_utterance.append(char)

    # breakpoint()

    return [
        {"display": "block"},
        speaker,
        annotated_utterance,
        fig,
        {"display": "block"},
        global_fig,
        {"display": "block"},
        explanations["global"],
        classifications["emotions"][0][0],
        classifications["emotions"][0][0][0],
        classifications["emotions"][0][1][0],
        classifications["emotions"][0][0][1],
        classifications["emotions"][0][1][1],
        classifications["emotions"][0][0][2],
        classifications["emotions"][0][1][2],
        {"display": "block"},
        json.dumps(empathy, indent=2),
    ]
