"""
Module Doc String
"""


import requests
from dash import callback, Input, Output, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from emotions.server.components.dialogue_dropdown import (
    textbox_popover,
    mi_data,
)

from emotions.server.callbacks import (
    build_utterance_component,
    build_micromodel_component,
)
from emotions.constants import FEATURIZER_SERVER
from emotions.server.init_server import CACHE
from emotions.constants import THERAPIST


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
    print("Display_dialogue.")
    print("dialogue_id:", dialogue_id)
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

    print("Display dialogue Returning...")

    analyses = []
    for idx, encoding in enumerate(dialogue_encoding):
        utterance_obj = mi_data[dialogue_id][idx]
        utterance = utterance_obj["utterance"]
        speaker = utterance_obj["speaker"]
        utterance_encoding = dialogue_encoding[idx]["results"]

        if speaker == THERAPIST:
            active_tab = "utterance-tab-1"
        else:
            active_tab = "utterance-tab-2"

        (
            tab_component,
            formatted_speaker,
            annotated_utterance,
        ) = build_utterance_component(
            utterance_obj, utterance_encoding, active_tab, idx
        )

        # if triggered_id == "micromodel-results":
        #    if hover_data is None:
        #        raise PreventUpdate
        #    data = hover_data["points"][0]
        #    micromodel = data["label"]
        #    query = " ".join(word_tokenize(data["customdata"][1]))
        #    segment = " ".join(word_tokenize(data["customdata"][0]))

        #    if segment == "":
        #        raise PreventUpdate

        #    else:
        #        segment_start_idx = query.index(segment)
        #        segment_end_idx = segment_start_idx + len(segment)

        #        annotated_utterance = (
        #            [query[:segment_start_idx]]
        #            + [
        #                entity(
        #                    query[segment_start_idx:segment_end_idx],
        #                    micromodel,
        #                    get_mm_color(micromodel),
        #                )
        #            ]
        #            + [query[segment_end_idx:]]
        #        )

        micromodel_component = build_micromodel_component(
            utterance_encoding["micromodels"], utterance, speaker, idx
        )

        _utterance_component = dbc.Card(
            [
                dbc.CardHeader(tab_component),
                dbc.CardBody(
                    [
                        html.H5(formatted_speaker),
                        html.Br(),
                        html.Div(
                            annotated_utterance,
                            id={
                                "type": "annotated-utterance-container",
                                "index": idx,
                            },
                        ),
                    ]
                ),
            ]
        )

        analysis = html.Div(
            children=[
                html.Div(
                    [
                        _utterance_component,
                        html.Br(),
                        micromodel_component,
                    ],
                    style={"display": "block"},
                ),
            ],
        )
        analyses.append(analysis)

    assert len(analyses) == len(mi_data[dialogue_id])
    return [
        [
            textbox_popover(
                utt_obj["utterance"],
                utt_obj["speaker"],
                idx,
                "dialogue-click",
                analyses[idx],
            )
            for idx, utt_obj in enumerate(mi_data[dialogue_id])
        ],
        dialogue_encoding,
    ]
