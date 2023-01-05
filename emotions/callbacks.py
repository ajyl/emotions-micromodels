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

from constants import FEATURIZER_SERVER, MITI_THRESHOLD, THERAPIST, PATIENT

COLOR_SCHEME = px.colors.qualitative.Set2


def _get_mm_color(mm_name):
    """
    Get color for mm.
    """
    mm_prefixes = ["emotion", "custom", "empathy", "miti"]
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
            annotated_utterance.extend(utterance[last_idx: idx_obj[1]])
            annotated_utterance.append(
                entity(
                    utterance[idx_obj[1] : idx_obj[2]],
                    idx_obj[0],
                    _get_mm_color(annotation_type),
                )
            )
            last_idx = idx_obj[2]
        annotated_utterance.extend(utterance[last_idx:])
    return annotated_utterance


def update_annotate_utterance(annotation_obj, tab_id):
    """
    Update utterance annotation.
    """
    annotation_type = {
        "utterance-tab-1": "miti",
        "utterance-tab-2": "custom_emotions",
        "utterance-tab-3": "miti",
        "utterance-tab-4": "empathy",
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
        ]

    triggered_id = ctx.triggered_id
    print(triggered_id)

    if triggered_id is None:
        raise PreventUpdate

    if triggered_id == "micromodel-results":
        return handle_hover(hover_data)

    if triggered_id == "utterance-tabs":
        return update_annotate_utterance(
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

    # MITI
    miti_results = {
        x: y
        for x, y in response_obj["micromodels"].items()
        if x.startswith("miti_")
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

    # MITI - PAIR
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
                word_tokenize(
                    response_obj["micromodels"][miti_code[0]]["segment"]
                )
            ),
        )
        for miti_code in _miti
    ]

    custom_emotion_segments = [
        (x, " ".join(word_tokenize(response_obj["micromodels"][x]["segment"])))
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
    miti_idxs = sorted(miti_idxs, key=lambda x: x[1])

    custom_emotion_idxs = [
        (
            emotion.replace("custom_", ""),
            _query.index(_segment),
            _query.index(_segment) + len(_segment),
            response_obj["micromodels"][emotion]["max_score"],
        )
        for emotion, _segment in custom_emotion_segments
    ]
    custom_emotion_idxs_2 = sorted(custom_emotion_idxs, key=lambda x: x[1])
    custom_emotion_idxs_3 = []

    prev = custom_emotion_idxs_2[0]
    for idx_obj in custom_emotion_idxs_2[1:]:
        curr_min = prev[1]
        curr_max = prev[2]
        curr_score = prev[3]

        new_min = idx_obj[1]
        new_max = idx_obj[2]
        new_score = idx_obj[3]

        # Overlap
        if curr_min <= new_min <= curr_max:
            if new_score > curr_score:
                prev = idx_obj

        # No Overlap
        elif new_min > curr_max:
            custom_emotion_idxs_3.append(prev)
            prev = idx_obj

        else:
            breakpoint()

    if prev not in custom_emotion_idxs_3:
        custom_emotion_idxs_3.append(prev)

    utterance_annotation_obj = {
        "utterance": _query,
        "miti": miti_idxs,
        "custom_emotions": custom_emotion_idxs_3,
    }
    # TODO: Update second argument.
    annotated_utterance = annotate_utterance(utterance_annotation_obj, "miti")

    return [
        {"display": "block"},
        speaker,
        annotated_utterance,
        utterance_annotation_obj,
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
