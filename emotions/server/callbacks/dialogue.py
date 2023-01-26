"""
Module Doc String
"""


import requests
from dash import callback, Input, Output
from dash.exceptions import PreventUpdate
from emotions.server.components.dialogue_dropdown import textbox, mi_data
from emotions.constants import FEATURIZER_SERVER


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

    response = requests.post(
        FEATURIZER_SERVER + "/encode_convo",
        json={"convo_id": dialogue_id, "convo": mi_data[dialogue_id]},
    )
    dialogue_encoding = response.json()
    return [
        [
            textbox(utt_obj["utterance"], utt_obj["speaker"], idx)
            for idx, utt_obj in enumerate(mi_data[dialogue_id])
        ],
        dialogue_encoding,
    ]
