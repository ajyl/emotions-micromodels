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

from emotions.constants import (
    FEATURIZER_SERVER,
    THERAPIST,
    PATIENT,
    MITI_THRESHOLD,
    EMOTION_THRESHOLD,
    EMPATHY_THRESHOLD,
)
from emotions.server.utils import get_mm_color, entity, COLOR_SCHEME
from emotions.server.callbacks.annotate_utterance import (
    update_utterance_component,
    annotate_utterance,
    get_annotation_spans,
)
from emotions.server.callbacks.hover import handle_hover


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
        no_update,
    ]


@callback(
    [
        Output("analysis", "style"),
        Output("speaker", "children"),
        Output("utterance", "children"),
        Output("annotated-utterance-storage", "data"),
        Output("utterance-tabs", "active_tab"),
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
        Output("pair", "children"),
    ],
    [
        Input({"type": "dialogue-click", "index": ALL}, "n_clicks"),
        Input("micromodel-results", "hoverData"),
        Input("global-explanation-feature-dropdown", "value"),
        Input("global-explanation-storage", "data"),
        Input("emotion-classification-storage", "data"),
        Input("utterance-tabs", "active_tab"),
        Input("annotated-utterance-storage", "data"),
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
    utterance_tab,
    annotated_utterance_storage,
    utterances,
):

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
            no_update,
            no_update,
            no_update,
        ]

    triggered_id = ctx.triggered_id
    print(triggered_id)

    if triggered_id is None:
        raise PreventUpdate

    if triggered_id == "micromodel-results":
        return handle_hover(hover_data)

    if triggered_id == "utterance-tabs":
        return update_utterance_component(
            annotated_utterance_storage, utterance_tab
        )

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

    prev_utterance = None
    if speaker == THERAPIST and idx > 0:
        prev_utterance = utterances[idx - 1]["utterance"]

    response = requests.post(
        FEATURIZER_SERVER + "/encode",
        json={"response": utterance, "prompt": prev_utterance},
    )
    response_obj = response.json()

    # Emotions - Predictions
    emotion_classifications = response_obj["emotion"]["predictions"][0][0]
    emotion_scores = response_obj["emotion"]["predictions"][0][1]

    # Emotions - Explanations
    explanations = response_obj["emotion"]["explanations"]
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

    # Empathy
    empathy = [
        response_obj["empathy"]["empathy_emotional_reactions"],
        response_obj["empathy"]["empathy_explorations"],
        response_obj["empathy"]["empathy_interpretations"],
    ]

    # MITI (PAIR)
    pair_results = response_obj.get("pair")
    if pair_results:
        score = pair_results["score"][0]
        pair_pred = "No Reflection"
        if score >= 0.3:
            pair_pred = "Simple Reflection"
        if score >= 0.7:
            pair_pred = "Complex Reflection"
        pair_results = {"prediction": pair_pred, "score": score}

    # Micromodels
    mms = list(response_obj["micromodels"].keys())
    sorted_mms = (
        sorted([mm for mm in mms if mm.startswith("emotion_")])
        + sorted([mm for mm in mms if mm.startswith("custom_")])
        + sorted([mm for mm in mms if mm.startswith("empathy_")])
        + sorted([mm for mm in mms if mm.startswith("miti_")])
    )

    data = []
    for mm in sorted_mms:
        mm_result = response_obj["micromodels"][mm]
        data.append(
            (
                mm,
                utterance,
                max(mm_result["max_score"], 0),
                mm_result["top_k_scores"][0][0],
                mm_result["top_k_scores"][0][1],
                mm_result["segment"],
                get_mm_color(mm),
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

    utterance = " ".join(word_tokenize(utterance))
    utterance_annotation_obj = {
        "utterance": utterance,
        "miti": get_annotation_spans(
            utterance, response_obj, "miti_", MITI_THRESHOLD
        ),
        "emotions": get_annotation_spans(
            utterance, response_obj, "custom_", EMOTION_THRESHOLD
        ),
        "empathy": get_annotation_spans(
            utterance, response_obj, "empathy_", EMPATHY_THRESHOLD
        ),
    }
    annotated_utterance = annotate_utterance(
        utterance_annotation_obj, "emotions"
    )

    return [
        {"display": "block"},
        speaker,
        annotated_utterance,
        utterance_annotation_obj,
        "utterance-tab-1",
        fig,
        {"display": "block"},
        global_fig,
        {"display": "block"},
        explanations["global"],
        emotion_classifications,
        emotion_classifications[0],
        emotion_scores[0],
        emotion_classifications[1],
        emotion_scores[1],
        emotion_classifications[2],
        emotion_scores[2],
        {"display": "block"},
        json.dumps(empathy, indent=2),
        json.dumps(pair_results, indent=2),
    ]
