"""
Components for dialogue.
"""

import os
import json
from collections import OrderedDict
from dash import html, dcc
import dash_bootstrap_components as dbc
from emotions.constants import THERAPIST, PATIENT


MM_HOME = os.environ.get("MM_HOME")
APP_HOME = os.path.join(MM_HOME, "emotions")
MI_DATA_DIR = os.path.join(APP_HOME, "data/HighLowQualityCounseling/json")
ANNO_MI_DATAPATH = os.path.join(APP_HOME, "data/AnnoMI/anno_mi.json")


def init_mi_dialogue_vero(data_dir=MI_DATA_DIR):
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


def init_anno_mi_data(data_filepath=ANNO_MI_DATAPATH):
    with open(data_filepath, "r") as file_p:
        data = json.load(file_p)
    sorted_dict = OrderedDict()

    transcript_ids_high = [
        (x.replace("high_", ""), x)
        for x in data.keys()
        if x.startswith("high_")
    ]
    transcript_ids_low = [
        (x.replace("low_", ""), x) for x in data.keys() if x.startswith("low_")
    ]
    sorted_ids_high = sorted(transcript_ids_high, key=lambda x: int(x[0]))
    sorted_ids_low = sorted(transcript_ids_low, key=lambda x: int(x[0]))

    for idx, _id in enumerate(sorted_ids_high):
        sorted_dict["high_%d" % idx] = data[_id[1]]
    for idx, _id in enumerate(sorted_ids_low):
        sorted_dict["low_%d" % idx] = data[_id[1]]

    return sorted_dict


mi_data = init_anno_mi_data()
dialogue_dropdown = html.Div(
    [dcc.Dropdown(list(mi_data.keys()), id="mi-dropdown")]
)


def textbox(text, box, idx, component_id):
    """
    Textbox component.
    """

    style = {
        "max-width": "55%",
        "width": "max-content",
        "border-radius": "16px",
    }

    if box == PATIENT:
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        color = "primary"
        inverse = True

    elif box == THERAPIST:
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
                id={"type": component_id, "index": idx},
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


def textbox_popover(text, box, idx, component_id):
    """
    Textbox component.
    """

    style = {
        "max-width": "55%",
        "width": "max-content",
        "border-radius": "16px",
    }

    if box == PATIENT:
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        color = "primary"
        inverse = True

    elif box == THERAPIST:
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        color = "light"
        inverse = False

    else:
        raise ValueError("Incorrect option for 'box'.")

    return dbc.Card(
        [
            html.Div(
                dbc.Button(
                    text,
                    id={"type": component_id, "index": idx},
                    n_clicks=0,
                    color=color,
                    value={"speaker": box, "utterance": text},
                ),
                id=f"popover-div-wrapper-{idx}",
            ),
            dbc.Popover(
                dbc.PopoverBody(
                    f"Hmm, init", id={"type": "popover_body", "index": idx}
                ),
                id={"type": "popover", "index": idx},
                is_open=False,
                target=f"popover-div-wrapper-{idx}",
                style={"width": "1000px", "max-width": "1000px"},
            ),
        ],
        style=style,
        body=True,
        color=color,
        inverse=inverse,
        id="dialogue-textbox-card"
    )
