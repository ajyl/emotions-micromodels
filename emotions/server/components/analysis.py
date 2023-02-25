"""
Dash Components
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from emotions.config import EMOTIONS


def get_emotion_mms():
    return ["emotion_%s" % emotion for emotion in EMOTIONS] + [
        "fasttext_emotion_%s" % emotion for emotion in EMOTIONS
    ]


utterance_tabs = dbc.Tabs(
    [
        dbc.Tab(label="MITI Codes", tab_id="utterance-tab-1"),
        dbc.Tab(label="Emotions", tab_id="utterance-tab-2"),
        dbc.Tab(label="Empathy", tab_id="utterance-tab-3"),
    ],
    id="utterance-tabs",
    active_tab="utterance-tab-1",
)

utterance_tabs_card = dbc.CardHeader(utterance_tabs)

annotated_utterance_component = html.Div()
utterance_component = dbc.Card(
    [
        utterance_tabs_card,
        dbc.CardBody(
            [annotated_utterance_component],
        ),
    ]
)
utterance_component_container = html.Div(utterance_component)


micromodel_bar_graph = dcc.Graph(
    id="micromodel-results", style={"display": "none"}
)
micromodel_bar_graph_container = html.Div(micromodel_bar_graph)
micromodel_component = dbc.Card(
    [
        micromodel_bar_graph_container,
        dcc.Store(id="emotion-classification-storage"),
    ]
)

explanation_dropdown = dcc.Dropdown(
    id="global-explanation-feature-dropdown",
    options=["Overall"] + get_emotion_mms(),
    multi=False,
    value="Overall",
)
explanation_graph = html.Div(
    [
        dcc.Graph(
            id="global-explanation",
            style={"display": "none"},
        )
    ]
)


emotion_table = html.Table()
emotion_analysis = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Emotion Analysis"),
            emotion_table,
            html.Br(),
            html.H6("Explanations:"),
            dbc.Card(
                [
                    explanation_dropdown,
                    explanation_graph,
                ],
            ),
        ]
    )
)

analysis = html.Div(
    children=[
        html.Div(
            [
                dcc.Store(id="annotated-utterance-storage"),
                utterance_component_container,
                html.Br(),
                micromodel_component,
                # html.Br(),
                # emotion_analysis,
            ],
            style={"display": "none"},
            id="analysis",
        ),
    ],
)
