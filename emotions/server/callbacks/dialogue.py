"""
Module Doc String
"""


import requests
from dash import callback, Input, Output
from dash.exceptions import PreventUpdate
from emotions.server.components.dialogue_dropdown import textbox, mi_data
from emotions.constants import FEATURIZER_SERVER
from emotions.server.init_server import CACHE


@callback(
    [
        Output("display-conversation", "children"),
        Output("conversation-encoding", "data"),
    ],
    Input("mi-dropdown", "value"),
)
def display_dialogue(dialogue_id):
    """
    Display dialogue.
    """
    if dialogue_id is None:
        raise PreventUpdate

    if dialogue_id in CACHE:
        dialogue_encoding = CACHE[dialogue_id]

    else:
        response = requests.post(
            FEATURIZER_SERVER + "/encode_convo",
            json={"convo_id": dialogue_id, "convo": mi_data[dialogue_id]},
        )
        dialogue_encoding = response.json()
    return [
        [
            textbox(utt_obj["utterance"], utt_obj["speaker"], idx, "dialogue-click")
            for idx, utt_obj in enumerate(mi_data[dialogue_id])
        ],
        dialogue_encoding,
    ]
