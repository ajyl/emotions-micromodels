"""
Dash Server
"""

import sys
import time
import requests
from dash import Dash, html
import dash_bootstrap_components as dbc
from components import analysis, dialogue_dropdown, conversation
from callbacks.encode import encode
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


try:
    init_featurizer_server(FEATURIZER_SERVER)
except RuntimeError:
    sys.exit()


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
        html.Div([html.H1(children="Welcome!")]),
        dialogue_dropdown,
        dbc.Row([dbc.Col(conversation, width=4), dbc.Col(analysis, width=8)]),
    ],
    fluid=True,
)


if __name__ == "__main__":
    app.run_server(debug=True)
