"""
Demo for Dialogue.
"""

import os
import sys
import json
import pickle
import base64
import time
import requests
import pandas as pd
from nltk.tokenize import word_tokenize
from dash import Dash, html, dcc, Input, Output, State, dash_table, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
from interpret.glassbox.ebm.ebm import EBMExplanation


FEATURIZER_SERVER = "http://127.0.0.1:5000"
MITI_THRESHOLD = 0.5
MI_DATA_DIR = "./data/HighLowQualityCounseling/json"


def init_featurizer_server(server_addr):
    """
    Initialize featurizer server.
    """
    retries = 20
    tries = 0
    featurizer_ready = False
    while not featurizer_ready:
        response = requests.get(server_addr + "/ping")
        featurizer_ready = response.json().get("ready", False)
        if not response.ok or not featurizer_ready:
            tries += 1
            print(
                "Waiting for featurizer server... (%d / %d)" % (tries, retries)
            )
            time.sleep(1)

    if not featurizer_ready:
        raise RuntimeError("Could not connect to featurizer server!")


def init_mi_dialogue(data_dir=MI_DATA_DIR):
    """
    Initialize MI dialogue data.
    """
    mi_data = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(data_dir, filename)) as file_p:
            data = json.load(file_p)

        mi_data[filename.replace(".json", "")] = data
    return mi_data


try:
    init_featurizer_server(FEATURIZER_SERVER)
except RuntimeError:
    sys.exit()


mi_data = init_mi_dialogue()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


def textbox(text, box, idx):
    """
    Textbox component.
    """

    style = {
        "max-width": "55%",
        "width": "max-content",
        #"padding": "1px 1px",
        #"border-radius": "6px",
    }

    if box == "patient":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        color = "primary"
        inverse = True

    elif box == "therapist":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        color = "light"
        inverse = False

    else:
        raise ValueError("Incorrect option for 'box'.")

    return dbc.Card(
        [
            dbc.Button(
                text,
                id={"type": "dialogue-click", "index": idx},
                n_clicks=0,
                color=color,
                value=text,
            )
        ],
        style=style,
        body=True,
        color=color,
        inverse=inverse,
    )


conversation = html.Div(
    style={
        "max-width": "800px",
        "height": "70vh",
        "margin": 0,
        "overflow-y": "auto",
    },
    id="display-conversation",
)

analysis = html.Div(
    [
        dcc.Graph(id="micromodel-results"),
        html.Table(
            [
                html.Thead(
                    html.Tr([html.Th("Emotion"), html.Th("Confidence Score")])
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(id="emotion_1"),
                                html.Td(id="emotion_score_1"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(id="emotion_2"),
                                html.Td(id="emotion_score_2"),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td(id="emotion_3"),
                                html.Td(id="emotion_score_3"),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        html.Pre(id="empathy"),
    ]
)


app.layout = dbc.Container(
    [
        html.Div([html.H1(children="Welcome!")]),
        html.Div(
            [dcc.Dropdown(sorted(list(mi_data.keys())), id="mi-dropdown")]
        ),
        dbc.Row(
            [dbc.Col(conversation, width=6), dbc.Col(analysis, width=6)]
        ),
    ]
)


def _get_color(mm_name):
    """
    Get color for mm.
    """
    mm_prefixes = ["emotion_", "custom_", "empathy_", "miti_"]
    for idx, prefix in enumerate(mm_prefixes):
        if mm_name.startswith(prefix):
            return idx
    raise ValueError("Unknown MM %s!" % mm_name)


@app.callback(
    Output("display-conversation", "children"), Input("mi-dropdown", "value")
)
def display_dialogue(dialogue_id):
    """
    Display dialogue.
    """
    if dialogue_id is None:
        return

    return [
        textbox(utt_obj["utterance"], utt_obj["speaker"], idx)
        for idx, utt_obj in enumerate(mi_data[dialogue_id])
    ]


@app.callback(
    [
        Output("micromodel-results", "figure"),
        # Outputexplanation", "figure"),
        Output("emotion_1", "children"),
        Output("emotion_score_1", "children"),
        Output("emotion_2", "children"),
        Output("emotion_score_2", "children"),
        Output("emotion_3", "children"),
        Output("emotion_score_3", "children"),
        Output("empathy", "children"),
    ],
    Input({"type": "dialogue-click", "index": ALL}, "n_clicks"),
    State({"type": "dialogue-click", "index": ALL}, "value"),
)
def encode(query, utterances):
    print("query------------------")
    print(query)
    print("utterances------------")
    print(utterances)
    print("trigger------------")
    print(ctx.triggered_id)
    if ctx.triggered_id is None:
        raise PreventUpdate

    idx = ctx.triggered_id["index"]
    query = utterances[idx]
    response = requests.post(
        FEATURIZER_SERVER + "/encode", json={"query": query}
    )
    result = response.json()

    # Classifications
    classifications = result["classifications"]

    empathy = [
        classifications["empathy_emotional_reactions"],
        classifications["empathy_explorations"],
        classifications["empathy_interpretations"],
    ]

    # Explanations
    explanations = result["explanations"]
    global_exp = pickle.loads(base64.b64decode(explanations["global"]))
    global_fig = global_exp.visualize(0)

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
    if len(_miti) > 0:
        query = " ".join(word_tokenize(query))
        for _miti_code, _segment in segments:
            _idx = query.index(_segment)
            query = (
                query[:_idx]
                + "[[ %s %s ]]"
                % (_miti_code.upper().replace("_", "-"), _segment)
                + query[_idx + len(_segment) :]
            )

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
                query,
                max(mm_result["max_score"], 0),
                mm_result["top_k_scores"][0][0],
                mm_result["top_k_scores"][0][1],
                mm_result["segment"],
                _get_color(mm),
            )
        )

    data = pd.DataFrame(
        data=data,
        columns=[
            "mm_name",
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
        y="mm_name",
        color="color",
        hover_name="mm_name",
        hover_data=["mm_name", "score", "similar_segment", "similar_score"],
        custom_data=["similar_segment", "similar_score", "segment", "query"],
        # title=query,
        orientation="h",
        height=900
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout()

    return [
        fig,
        # global_fig,
        classifications["emotions"][0][0][0],
        classifications["emotions"][0][1][0],
        classifications["emotions"][0][0][1],
        classifications["emotions"][0][1][1],
        classifications["emotions"][0][0][2],
        classifications["emotions"][0][1][2],
        json.dumps(empathy, indent=2),
    ]


def main():
    """ Driver """
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
