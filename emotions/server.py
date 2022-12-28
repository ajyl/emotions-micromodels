"""
Server Module
"""

import sys
import json
import pickle
import base64
import time
import requests
import pandas as pd
from nltk.tokenize import word_tokenize
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from interpret.glassbox.ebm.ebm import EBMExplanation


FEATURIZER_SERVER = "http://127.0.0.1:5000"

styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}
MITI_THRESHOLD = 0.5


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


try:
    init_featurizer_server(FEATURIZER_SERVER)
except RuntimeError:
    sys.exit()


app = Dash(__name__)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div(
                    children=[
                        html.H1(children="Welcome!"),
                        html.Div(
                            children=[
                                html.Div(
                                    [
                                        "Input Query:",
                                        dcc.Input(
                                            id="query",
                                            value="I almost got into a car accident.",
                                            type="text",
                                        ),
                                    ]
                                ),
                                html.Button("Run!", id="submit", n_clicks=0),
                                html.Br(),
                                html.Div(id="query-output"),
                                dcc.Graph(id="results"),
                            ]
                        ),
                    ]
                )
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        children=[
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("Emotion"),
                                                html.Th("Confidence Score"),
                                            ]
                                        )
                                    ),
                                    html.Tbody(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td(id="emotion_1"),
                                                    html.Td(
                                                        id="emotion_score_1"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td(id="emotion_2"),
                                                    html.Td(
                                                        id="emotion_score_2"
                                                    ),
                                                ]
                                            ),
                                            html.Tr(
                                                [
                                                    html.Td(id="emotion_3"),
                                                    html.Td(
                                                        id="emotion_score_3"
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                    ),
                ),
                dbc.Col(
                    html.Div(
                        children=[
                            dcc.Graph(id="global_explanation"),
                            html.Div(
                                children=[
                                    html.Div(
                                        [
                                            html.Pre(
                                                id="hover-data",
                                                style=styles["pre"],
                                            )
                                        ],
                                        #className="three columns",
                                    )
                                ],
                            ),
                        ],
                    ),
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                html.Div(children=[html.Pre(id="empathy")]),
            )
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
    [
        Output(component_id="results", component_property="figure"),
        Output(component_id="global_explanation", component_property="figure"),
        Output(component_id="emotion_1", component_property="children"),
        Output(component_id="emotion_score_1", component_property="children"),
        Output(component_id="emotion_2", component_property="children"),
        Output(component_id="emotion_score_2", component_property="children"),
        Output(component_id="emotion_3", component_property="children"),
        Output(component_id="emotion_score_3", component_property="children"),
        Output(component_id="empathy", component_property="children"),
    ],
    Input("submit", "n_clicks"),
    State(component_id="query", component_property="value"),
)
def encode(n_clicks, query):
    response = requests.post(
        FEATURIZER_SERVER + "/encode", json={"query": query}
    )
    result = response.json()

    # Classifications
    classifications = result["classifications"]
    emotion_classification = pd.DataFrame(
        {
            "Emotion": classifications["emotions"][0][0][:3],
            "Confidence Scores": classifications["emotions"][0][1][:3],
        }
    )

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
        x="mm_name",
        y="score",
        color="color",
        hover_name="mm_name",
        hover_data=["mm_name", "score", "similar_segment", "similar_score"],
        custom_data=["similar_segment", "similar_score", "segment", "query"],
        #title=query,
    )
    fig.update_layout()


    return [
        fig,
        global_fig,
        classifications["emotions"][0][0][0],
        classifications["emotions"][0][1][0],
        classifications["emotions"][0][0][1],
        classifications["emotions"][0][1][1],
        classifications["emotions"][0][0][2],
        classifications["emotions"][0][1][2],
        json.dumps(empathy, indent=2),
    ]


@app.callback(
    [
        Output("query-output", "children"),
        Output("hover-data", "children"),
    ],
    Input("results", "hoverData"),
)
def display_hover_data(hoverData):
    if hoverData is None:
        return (None, None)

    print(json.dumps(hoverData, indent=2))
    data = hoverData["points"][0]
    info = {
        "micromodel": data["label"],
        "score": data["value"],
        "similar_segment": data["customdata"][0],
        "similar_score": data["customdata"][1],
        "segment": data["customdata"][2],
    }

    query = " ".join(word_tokenize(data["customdata"][3]))
    segment = " ".join(word_tokenize(data["customdata"][2]))

    segment_start_idx = query.index(segment)
    segment_end_idx = segment_start_idx + len(segment)

    annotated_query = []
    for idx, char in enumerate(query):
        if idx >= segment_start_idx and idx <= segment_end_idx:
            annotated_query.append(html.Span(char, className="span-text"))
        else:
            annotated_query.append(char)
    
    return [annotated_query, json.dumps(info, indent=2)]


def main():
    """ Driver """
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
