"""
Initialization functions for server.
"""

import requests
import json
import pickle
import time
import sys
import base64
from dash import Dash, html
import dash_bootstrap_components as dbc
from interpret.glassbox.ebm.ebm import EBMExplanation
from emotions.server.components import (
    analysis,
    dialogue_dropdown,
    conversation,
)
from emotions.constants import FEATURIZER_SERVER


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


def init_explanation(server_addr):
    response = requests.get(server_addr + "/explain")
    if not response.ok:
        raise RuntimeError("Could not fetch explanation")

    return pickle.loads(
        base64.b64decode(response.json()["explanations"]["global"])
    )


try:
    init_featurizer_server(FEATURIZER_SERVER)
except RuntimeError:
    sys.exit()


EMOTION_EXPL = init_explanation(FEATURIZER_SERVER)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
        html.Div([html.H1(children="Welcome!")]),
        dialogue_dropdown,
        dbc.Row([dbc.Col(conversation, width=4), dbc.Col(analysis, width=8)]),
    ],
    fluid=True,
)
server = app.server
