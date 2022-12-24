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
from dash import Dash, html, dcc, Input, Output, State
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
app.layout = html.Div(
    children=[
        html.H1(children="Welcome!"),
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
        dcc.Graph(id="results"),
        html.Div(
            children=[html.Pre(id="classifications")],
        ),
        dcc.Graph(id="global_explanation"),
        html.Div(
            children=[
                html.Div(
                    [html.Pre(id="hover-data", style=styles["pre"])],
                    className="three columns",
                )
            ],
        ),
        html.Div(children=[html.Pre(id="empathy")]),
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
        Output(component_id="classifications", component_property="children"),
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
    emotion_classification = list(
        zip(
            classifications["emotions"][0][0],
            classifications["emotions"][0][1],
        )
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
                max(mm_result["max_score"], 0),
                mm_result["top_k_scores"][0][0],
                mm_result["top_k_scores"][0][1],
                _get_color(mm),
            )
        )

    data = pd.DataFrame(
        data=data,
        columns=[
            "mm_name",
            "score",
            "similar_segment",
            "similar_score",
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
        custom_data=["similar_segment", "similar_score"],
        title=query,
    )
    fig.update_layout()

    return [
        fig,
        global_fig,
        json.dumps(emotion_classification[:3]),
        json.dumps(empathy, indent=2),
    ]


@app.callback(
    Output("hover-data", "children"),
    Input("results", "hoverData"),
)
def display_hover_data(hoverData):
    if hoverData is None:
        return

    print(json.dumps(hoverData, indent=2))
    data = hoverData["points"][0]
    info = {
        "micromodel": data["label"],
        "score": data["value"],
        "similar_segment": data["customdata"][0],
        "similar_score": data["customdata"][1],
    }
    return json.dumps(info, indent=2)


def main():
    """ Driver """
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
