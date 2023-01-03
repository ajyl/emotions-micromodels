"""
Components for dialogue.
"""

import os
import json
from dash import html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


MM_HOME = os.environ.get("MM_HOME")
APP_HOME = os.path.join(MM_HOME, "emotions")
MI_DATA_DIR = os.path.join(APP_HOME, "data/HighLowQualityCounseling/json")


def init_mi_dialogue(data_dir=MI_DATA_DIR):
    """
    Initialize MI dialogue data.
    """
    _mi_data = {}
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(data_dir, filename)) as file_p:
            data = json.load(file_p)

        _mi_data[filename.replace(".json", "")] = data
    return _mi_data


mi_data = init_mi_dialogue()
dialogue_dropdown = html.Div(
    [dcc.Dropdown(sorted(list(mi_data.keys())), id="mi-dropdown")]
)


def textbox(text, box, idx):
    """
    Textbox component.
    """

    style = {
        "max-width": "55%",
        "width": "max-content",
        # "padding": "1px 1px",
        # "border-radius": "6px",
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
                value={"speaker": box, "utterance": text},
            )
        ],
        style=style,
        body=True,
        color=color,
        inverse=inverse,
    )


@callback(
    Output("display-conversation", "children"),
    Input("mi-dropdown", "value"),
)
def display_dialogue(dialogue_id):
    """
    Display dialogue.
    """
    if dialogue_id is None:
        raise PreventUpdate

    return [
        textbox(utt_obj["utterance"], utt_obj["speaker"], idx)
        for idx, utt_obj in enumerate(mi_data[dialogue_id])
    ]
